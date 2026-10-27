/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 3: Streaming / paging.
 *
 *  Cluster data is serialized as *self-contained pages*: each page carries
 *  the vertex/index data of its cluster range (with page-local vertex ids)
 *  plus the cluster records, so a page can be loaded from disk with a single
 *  seek + read without touching any other page.
 *
 *  The page manager keeps a bounded residency pool, loads requested pages on
 *  a background worker thread (async disk I/O) and assembles the resident
 *  subset into a regular #TesseraData every time the resident set changes.
 *  Non-resident clusters stay in the assembled data as stubs (indexCount ==
 *  0, flagged); LOD selection treats them as "not drawable" and emits the
 *  streaming requests for what would have been drawn instead
 *  (error-driven requests).
 */

#pragma once

#include "tessera.h"
#include "tessera_lod.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace blender {
namespace tessera {

/** Clusters per streaming page. */
constexpr uint32_t kTesseraClustersPerPage = 128;

/** Cluster flag: the cluster's vertex/index data is not resident. */
constexpr uint32_t kTesseraClusterFlagNotResident = 1u << 2;

/** One entry of the page directory (20 bytes). */
struct TesseraPageDirEntry {
    uint32_t clusterBegin;
    uint32_t clusterCount;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t byteSize;
};

/** Decoded contents of one page. */
struct TesseraPageContents {
    std::vector<float> positions;   /* 3 per vertex */
    std::vector<float> normals;     /* 3 per vertex */
    std::vector<float> uvs;         /* 2 per vertex */
    std::vector<uint32_t> indices;  /* page-local */
    std::vector<TesseraCluster> clusters;
};

/** Live streaming telemetry and memory budgeting state. */
struct TesseraMemoryBudget {
    size_t max_vram_bytes = 1024ULL * 1024ULL * 1024ULL; /* 1 GB default */
    size_t current_resident_bytes = 0;
    size_t pending_fetch_bytes = 0;
    uint32_t total_instances_in_scene = 0;
    uint32_t active_instances_rendered = 0;
    float disk_read_mbs = 0.0f;
    float cache_hit_ratio = 100.0f;
};

/**
 * Serialize a #TesseraData into the self-contained paged format (version 2).
 * Shared vertices are duplicated per page so pages stay independent.
 */
bool serializePagedToMemory(const TesseraData &data,
                            uint32_t clustersPerPage,
                            std::vector<uint8_t> &out);

/**
 * Decode one page from a paged file image. \param dir is the page's
 * directory entry; \param page_data points at the page block.
 */
bool decodePage(const TesseraPageDirEntry &dir,
                const uint8_t *page_data,
                size_t available,
                TesseraPageContents &out);

/**
 * Streaming page manager with async disk I/O.
 *
 * Usage (main thread):
 *   manager.open(path);
 *   loop {
 *     manager.requestClusters(requests);       // error-driven requests
 *     manager.update(frame, assembled);        // drains completed loads
 *     draw(assembled);                         // stubs for missing pages
 *   }
 */
class TesseraPageManager
{
 public:
    TesseraPageManager() = default;
    ~TesseraPageManager();

    TesseraPageManager(const TesseraPageManager&) = delete;
    TesseraPageManager& operator=(const TesseraPageManager&) = delete;

    /** Open a paged .nc2 file (v2 format) and start the worker thread. */
    bool open(const std::string &path);

    /** Stop the worker thread and close the file. */
    void close();

    bool isOpen() const { return mFile != nullptr; }
    uint32_t pageCount() const { return uint32_t(mDir.size()); }
    uint32_t clusterCount() const { return mHeader.clusterCount; }
    uint64_t residentBytes() const { return mResidentBytes; }

    /** Residency byte budget (eviction target). Default: 1024 MiB. */
    void setBudget(uint64_t bytes);
    uint64_t getBudget() const;

    /** Get live memory budgeting & streaming telemetry. */
    TesseraMemoryBudget getMemoryBudget() const;

    /** Pre-fetch clusters using camera trajectory & lookahead horizon. */
    void prefetchClusters(const std::vector<uint32_t> &cluster_ids);
    void prefetchFrustum(const float4x4 &viewProj,
                         float projScaleY,
                         float viewportHeight,
                         float lookaheadMargin,
                         const float3 &camPos,
                         const float3 &camVelocity);

    /** Update active & total scene instance telemetry. */
    void setInstanceCounts(uint32_t total_instances, uint32_t active_instances);

    /** True when every cluster's page is resident. */
    bool fullyResident() const;

    /** Request the pages of the given clusters (thread-safe). */
    void requestClusters(const std::vector<uint32_t> &cluster_ids);

    /** Request a page directly (thread-safe). */
    void requestPage(uint32_t page);

    /**
     * Provide the full cluster record table (identity, links, errors,
     * spheres). Metadata is tiny and always kept resident (like UE's node
     * table); only the vertex/index payload streams. Call after building or
     * fully loading the data.
     */
    void seedRecords(const TesseraData &full);

    /**
     * Main-thread pump: incorporate completed loads, evict over budget and
     * rebuild \a out_assembled when the resident set changed. \returns true
     * when the assembled data was modified (caller should re-upload).
     */
    bool update(uint64_t frame, TesseraData &out_assembled);

    /** Residency query for tests / statistics. */
    bool isPageResident(uint32_t page) const;

 private:
    void workerLoop();
    bool loadPageFromDisk(uint32_t page, TesseraPageContents &out);

    // Format data (read once in open()).
    TesseraHeader mHeader = {};
    std::vector<TesseraPageDirEntry> mDir;

    // File handle (all reads serialized by mIoMutex).
    std::FILE *mFile = nullptr;
    std::mutex mIoMutex;

    // Full cluster record table (identity/links/errors), always resident.
    std::vector<TesseraCluster> mRecordCache;

    // Residency state (guarded by mStateMutex).
    struct ResidentPage {
        TesseraPageContents contents;
        uint64_t last_use_frame = 0;
        uint64_t bytes = 0;
    };
    std::unordered_map<uint32_t, ResidentPage> mResident;
    uint64_t mResidentBytes = 0;
    uint64_t mBudgetBytes = 1024ULL * 1024ULL * 1024ULL;
    mutable std::mutex mStateMutex;

    mutable std::mutex mBudgetMutex;
    TesseraMemoryBudget mBudget;
    std::atomic<uint64_t> mTotalBytesRead{0};
    std::atomic<uint64_t> mTotalRequests{0};
    std::atomic<uint64_t> mCacheHits{0};
    std::atomic<uint64_t> mPendingBytes{0};
    std::chrono::steady_clock::time_point mIoStartTime = std::chrono::steady_clock::now();

    // Request queue (guarded by mWorkMutex + signaled via mWorkCondVar).
    std::deque<uint32_t> mPending;
    std::unordered_map<uint32_t, uint64_t> mInFlight;
    mutable std::mutex mWorkMutex;
    std::condition_variable mWorkCondVar;
    bool mStop = false;

    // Completed loads (guarded by mDoneMutex).
    std::deque<std::pair<uint32_t, TesseraPageContents>> mCompleted;
    std::mutex mDoneMutex;

    std::thread mWorker;
    bool mAssembledDirty = false;
};

/**
 * Error-driven request collection: walk the flat LOD rule ignoring the
 * residency of the candidate cluster itself and emit the cluster ids that
 * would be drawn but are stubs (indexCount == 0). The caller feeds the
 * result to #TesseraPageManager::requestClusters.
 */
void collectStreamingRequests(const TesseraData &data,
                              const float4x4 &viewProj,
                              float projScaleY,
                              float viewportHeight,
                              const TesseraLodParams &params,
                              std::vector<uint32_t> &out_requests);

}  // namespace tessera
}  // namespace blender