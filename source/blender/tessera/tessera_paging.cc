/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 3: Streaming / paging implementation.
 */

#include "tessera_paging.hh"


#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace blender {
namespace tessera {

namespace {

constexpr uint32_t kNc2VersionPaged = 2;

struct PageBuild {
    TesseraPageDirEntry dir = {};
    std::vector<uint32_t> vert_ids;      /* global ids, in page-local order */
    std::vector<uint32_t> indices;       /* page-local */
    std::vector<TesseraCluster> clusters; /* global records */
};

}  // namespace

bool serializePagedToMemory(const TesseraData &data,
                            uint32_t clustersPerPage,
                            std::vector<uint8_t> &out)
{
    if (data.clusters.empty() || clustersPerPage == 0) {
        return false;
    }

    /* Split the cluster array into contiguous pages. Indices are assigned
     * contiguously per cluster, so a page's index range is contiguous as
     * well; each page keeps the (duplicated) vertices its clusters need. */
    const uint32_t cluster_total = uint32_t(data.clusters.size());
    const uint32_t page_total =
        uint32_t((cluster_total + clustersPerPage - 1) / clustersPerPage);

    std::vector<PageBuild> pages(page_total);

    for (uint32_t p = 0; p < page_total; ++p) {
        PageBuild &page = pages[p];
        page.dir.clusterBegin = p * clustersPerPage;
        page.dir.clusterCount = std::min(clustersPerPage,
                                         cluster_total - page.dir.clusterBegin);

        /* Index range of the page (contiguous by construction) and the
         * vertices its clusters reference. */
        std::vector<uint32_t> referenced;
        for (uint32_t c = 0; c < page.dir.clusterCount; ++c) {
            const uint32_t ci = page.dir.clusterBegin + c;
            const TesseraCluster &nc = data.clusters[ci];
            for (uint32_t k = 0; k < nc.indexCount; ++k) {
                referenced.push_back(data.indices[nc.firstIndex + k]);
            }
            page.clusters.push_back(nc);
        }
        page.dir.indexCount = uint32_t(referenced.size());

        /* Compact the referenced vertices into page-local ids. */
        std::unordered_map<uint32_t, uint32_t> remap;
        page.dir.vertexCount = 0;
        for (const uint32_t g : referenced) {
            auto it = remap.find(g);
            if (it != remap.end()) {
                page.indices.push_back(it->second);
                continue;
            }
            const uint32_t local = page.dir.vertexCount;
            remap.emplace(g, local);
            page.vert_ids.push_back(g);
            page.dir.vertexCount++;
            page.indices.push_back(local);
        }

        /* Byte size: vertices (32 B) + indices (4 B) + clusters (72 B). */
        page.dir.byteSize = page.dir.vertexCount * 32 +
                            page.dir.indexCount * 4 +
                            page.dir.clusterCount * sizeof(TesseraCluster);
    }

    /* Emit: header + directory + pages. */
    const size_t header_size = sizeof(TesseraHeader);
    const size_t dir_size = size_t(page_total) * sizeof(TesseraPageDirEntry);
    size_t payload = 0;
    for (const PageBuild &page : pages) {
        payload += page.dir.byteSize;
    }
    out.resize(header_size + dir_size + payload);

    uint8_t *ptr = out.data();

    TesseraHeader header = data.header;
    header.version = kNc2VersionPaged;
    header.pageCount = page_total;
    std::memcpy(ptr, &header, header_size);
    ptr += header_size;

    for (const PageBuild &page : pages) {
        std::memcpy(ptr, &page.dir, sizeof(TesseraPageDirEntry));
        ptr += sizeof(TesseraPageDirEntry);
    }

    for (const PageBuild &page : pages) {
        for (const uint32_t g : page.vert_ids) {
            float pos[3] = {0.0f, 0.0f, 0.0f};
            float nrm[3] = {0.0f, 0.0f, 0.0f};
            float uv[2] = {0.0f, 0.0f};
            if (g < data.positions.size()) {
                const float3 &p = data.positions[g];
                pos[0] = p.x;
                pos[1] = p.y;
                pos[2] = p.z;
            }
            if (g < data.normals.size()) {
                const float3 &n = data.normals[g];
                nrm[0] = n.x;
                nrm[1] = n.y;
                nrm[2] = n.z;
            }
            if (g < data.uvs.size()) {
                const float2 &t = data.uvs[g];
                uv[0] = t.x;
                uv[1] = t.y;
            }
            std::memcpy(ptr, pos, 12);
            std::memcpy(ptr + 12, nrm, 12);
            std::memcpy(ptr + 24, uv, 8);
            ptr += 32;
        }
        if (page.dir.indexCount > 0) {
            std::memcpy(ptr,
                        page.indices.data(),
                        size_t(page.dir.indexCount) * sizeof(uint32_t));
            ptr += size_t(page.dir.indexCount) * sizeof(uint32_t);
        }
        std::memcpy(ptr,
                    page.clusters.data(),
                    size_t(page.dir.clusterCount) * sizeof(TesseraCluster));
        ptr += size_t(page.dir.clusterCount) * sizeof(TesseraCluster);
    }
    return true;
}

bool decodePage(const TesseraPageDirEntry &dir,
                const uint8_t *page_data,
                size_t available,
                TesseraPageContents &out)
{
    const size_t need = size_t(dir.vertexCount) * 32 +
                        size_t(dir.indexCount) * 4 +
                        size_t(dir.clusterCount) * sizeof(TesseraCluster);
    if (available < need || dir.byteSize != need) {
        return false;
    }

    const uint8_t *ptr = page_data;
    out.positions.resize(size_t(dir.vertexCount) * 3);
    out.normals.resize(size_t(dir.vertexCount) * 3);
    out.uvs.resize(size_t(dir.vertexCount) * 2);
    for (uint32_t v = 0; v < dir.vertexCount; ++v) {
        std::memcpy(&out.positions[v * 3], ptr, 12);
        std::memcpy(&out.normals[v * 3], ptr + 12, 12);
        std::memcpy(&out.uvs[v * 2], ptr + 24, 8);
        ptr += 32;
    }
    out.indices.resize(dir.indexCount);
    if (dir.indexCount > 0) {
        std::memcpy(out.indices.data(), ptr, size_t(dir.indexCount) * 4);
        ptr += size_t(dir.indexCount) * 4;
    }
    out.clusters.resize(dir.clusterCount);
    std::memcpy(out.clusters.data(),
                ptr,
                size_t(dir.clusterCount) * sizeof(TesseraCluster));
    return true;
}

/* ---------------------------------------------------------------------------
 * Page manager.
 * ------------------------------------------------------------------------- */

TesseraPageManager::~TesseraPageManager()
{
    close();
}

static uint64_t pageFileOffset(const TesseraHeader &header,
                               const std::vector<TesseraPageDirEntry> &dir,
                               uint32_t page)
{
    uint64_t offset = sizeof(TesseraHeader) +
                      size_t(header.pageCount) * sizeof(TesseraPageDirEntry);
    for (uint32_t p = 0; p < page; ++p) {
        offset += dir[p].byteSize;
    }
    return offset;
}

bool TesseraPageManager::open(const std::string &path)
{
    close();

    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }

    TesseraHeader header = {};
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != kNc2Magic || header.version != kNc2VersionPaged ||
        header.pageCount == 0)
    {
        fclose(file);
        return false;
    }

    std::vector<TesseraPageDirEntry> dir(header.pageCount);
    if (fread(dir.data(), sizeof(TesseraPageDirEntry), dir.size(), file) !=
        dir.size())
    {
        fclose(file);
        return false;
    }

    mHeader = header;
    mDir = std::move(dir);
    mFile = file;
    mStop = false;
    mWorker = std::thread([this]() { workerLoop(); });
    return true;
}

void TesseraPageManager::close()
{
    if (mWorker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mWorkMutex);
            mStop = true;
        }
        mWorkCondVar.notify_all();
        mWorker.join();
    }
    if (mFile != nullptr) {
        std::fclose(mFile);
        mFile = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(mStateMutex);
        mResident.clear();
        mResidentBytes = 0;
    }
    {
        std::lock_guard<std::mutex> lock(mWorkMutex);
        mPending.clear();
        mInFlight.clear();
    }
    {
        std::lock_guard<std::mutex> lock(mDoneMutex);
        mCompleted.clear();
    }
}

bool TesseraPageManager::isPageResident(uint32_t page) const
{
    std::lock_guard<std::mutex> lock(mStateMutex);
    return mResident.count(page) != 0;
}

bool TesseraPageManager::fullyResident() const
{
    std::lock_guard<std::mutex> lock(mStateMutex);
    return mResident.size() == size_t(mHeader.pageCount);
}

void TesseraPageManager::requestPage(uint32_t page)
{
    if (page >= mHeader.pageCount) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mWorkMutex);
        if (mStop || mInFlight.count(page) != 0 ||
            mResident.count(page) != 0)
        {
            return;
        }
        mPending.push_back(page);
        mInFlight.emplace(page, 0);
    }
    mWorkCondVar.notify_one();
}

void TesseraPageManager::requestClusters(const std::vector<uint32_t> &cluster_ids)
{
    for (const uint32_t cid : cluster_ids) {
        requestPage(cid / kTesseraClustersPerPage);
    }
}

bool TesseraPageManager::loadPageFromDisk(uint32_t page,
                                         TesseraPageContents &out)
{
    const TesseraPageDirEntry &entry = mDir[page];
    std::vector<uint8_t> buffer(entry.byteSize);

    {
        std::lock_guard<std::mutex> lock(mIoMutex);
        if (mFile == nullptr) {
            return false;
        }
        const uint64_t offset = pageFileOffset(mHeader, mDir, page);
#if defined(_WIN32)
        if (_fseeki64(mFile, int64_t(offset), SEEK_SET) != 0) {
            return false;
        }
#else
        if (fseeko(mFile, off_t(offset), SEEK_SET) != 0) {
            return false;
        }
#endif
        if (fread(buffer.data(), 1, buffer.size(), mFile) != buffer.size()) {
            return false;
        }
    }
    return decodePage(entry, buffer.data(), buffer.size(), out);
}

void TesseraPageManager::workerLoop()
{
    for (;;) {
        uint32_t page;
        {
            std::unique_lock<std::mutex> lock(mWorkMutex);
            mWorkCondVar.wait(lock, [this]() {
                return mStop || !mPending.empty();
            });
            if (mStop && mPending.empty()) {
                return;
            }
            page = mPending.front();
            mPending.pop_front();
        }

        TesseraPageContents contents;
        loadPageFromDisk(page, contents);

        {
            std::lock_guard<std::mutex> lock(mDoneMutex);
            mCompleted.emplace_back(page, std::move(contents));
        }
        /* Mark in-flight as finished so future requests can re-queue the
         * page if it was evicted before completion. */
        {
            std::lock_guard<std::mutex> lock(mWorkMutex);
            mInFlight.erase(page);
        }
    }
}

void TesseraPageManager::seedRecords(const TesseraData &full)
{
    mRecordCache = full.clusters;
    mAssembledDirty = true;
}

bool TesseraPageManager::update(uint64_t frame, TesseraData &out_assembled)
{
    bool changed = false;

    /* Incorporate completed loads. */
    for (;;) {
        std::pair<uint32_t, TesseraPageContents> item;
        {
            std::lock_guard<std::mutex> lock(mDoneMutex);
            if (mCompleted.empty()) {
                break;
            }
            item = std::move(mCompleted.front());
            mCompleted.pop_front();
        }
        const uint32_t page = item.first;
        const TesseraPageContents &contents = item.second;
        const TesseraPageDirEntry &entry = mDir[page];
        if (contents.positions.size() != size_t(entry.vertexCount) * 3) {
            continue; /* failed load; will be re-requested */
        }

        std::lock_guard<std::mutex> lock(mStateMutex);
        if (mResident.count(page) != 0) {
            continue;
        }
        ResidentPage &resident = mResident[page];
        resident.contents = std::move(item.second);
        resident.bytes = entry.byteSize;
        resident.last_use_frame = frame;
        mResidentBytes += entry.byteSize;
        changed = true;
    }

    /* Evict over budget: least-recently-used first, never pages used in the
     * last few frames. */
    {
        std::lock_guard<std::mutex> lock(mStateMutex);
        while (mResidentBytes > mBudgetBytes) {
            auto oldest = mResident.end();
            for (auto it = mResident.begin(); it != mResident.end(); ++it) {
                if (frame - it->second.last_use_frame < 8) {
                    continue;
                }
                if (oldest == mResident.end() ||
                    it->second.last_use_frame < oldest->second.last_use_frame)
                {
                    oldest = it;
                }
            }
            if (oldest == mResident.end()) {
                break; /* everything recently used */
            }
            mResidentBytes -= oldest->second.bytes;
            mResident.erase(oldest);
            changed = true;
        }
    }

    if (!changed && !mAssembledDirty) {
        return false;
    }

    /* Rebuild the assembled resident data:
     * - all cluster records are present in global order (from the record
     *   cache) so parent links stay valid;
     * - non-resident clusters become stubs: indexCount = 0 and the
     *   kTesseraClusterFlagNotResident flag set;
     * - vertex/index arrays hold only resident pages' payloads, in page
     *   order, with page-local ids remapped to assembled-global ids. */
    const uint32_t cluster_total = mHeader.clusterCount;

    std::vector<uint64_t> index_offsets(mHeader.pageCount + 1, 0);
    std::vector<uint64_t> vertex_offsets(mHeader.pageCount + 1, 0);
    {
        std::lock_guard<std::mutex> lock(mStateMutex);
        for (uint32_t p = 0; p < mHeader.pageCount; ++p) {
            auto it = mResident.find(p);
            if (it == mResident.end()) {
                continue;
            }
            index_offsets[p + 1] =
                index_offsets[p] + it->second.contents.indices.size();
            vertex_offsets[p + 1] =
                vertex_offsets[p] + it->second.contents.positions.size() / 3;
        }
    }

    out_assembled.header = mHeader;
    out_assembled.header.clusterCount = cluster_total;
    out_assembled.clusters.clear();
    out_assembled.clusters.reserve(cluster_total);
    out_assembled.indices.clear();
    out_assembled.positions.clear();
    out_assembled.normals.clear();
    out_assembled.uvs.clear();
    out_assembled.proxyPositions.clear();
    out_assembled.proxyIndices.clear();

    uint32_t level_count = 0;
    for (uint32_t p = 0; p < mHeader.pageCount; ++p) {
        TesseraPageContents *contents = nullptr;
        {
            std::lock_guard<std::mutex> lock(mStateMutex);
            auto it = mResident.find(p);
            if (it != mResident.end()) {
                contents = &it->second.contents;
            }
        }

        const uint32_t begin = mDir[p].clusterBegin;
        const uint32_t count = mDir[p].clusterCount;

        if (contents == nullptr) {
            /* Stub the page's clusters from the cached record table. */
            for (uint32_t c = 0; c < count; ++c) {
                TesseraCluster stub = {};
                if (begin + c < mRecordCache.size()) {
                    stub = mRecordCache[begin + c];
                }
                stub.indexCount = 0;
                stub.firstIndex = 0;
                stub.flags |= kTesseraClusterFlagNotResident;
                out_assembled.clusters.push_back(stub);
            }
            continue;
        }

        /* Append the page's vertices and remap its indices. */
        const uint64_t vertex_base = vertex_offsets[p];
        out_assembled.positions.insert(out_assembled.positions.end(),
                                       contents->positions.begin(),
                                       contents->positions.end());
        out_assembled.normals.insert(out_assembled.normals.end(),
                                     contents->normals.begin(),
                                     contents->normals.end());
        out_assembled.uvs.insert(
            out_assembled.uvs.end(), contents->uvs.begin(), contents->uvs.end());

        const uint64_t index_base = index_offsets[p];
        for (const uint32_t local : contents->indices) {
            out_assembled.indices.push_back(uint32_t(vertex_base + local));
        }

        for (uint32_t c = 0; c < count; ++c) {
            TesseraCluster nc = contents->clusters[c];
            nc.firstIndex = uint32_t(index_base + nc.firstIndex);
            nc.group = begin + c;
            if (begin + c < mRecordCache.size()) {
                /* Identity/link metadata comes from the always-resident
                 * record table (survives eviction). */
                const TesseraCluster &rec = mRecordCache[begin + c];
                nc.error = rec.error;
                nc.parentError = rec.parentError;
                nc.parentGroup = rec.parentGroup;
                nc.level = rec.level;
                std::memcpy(nc.sphere, rec.sphere, sizeof(nc.sphere));
                std::memcpy(
                    nc.parentSphere, rec.parentSphere, sizeof(nc.parentSphere));
            }
            level_count = std::max(level_count, nc.level + 1);
            out_assembled.clusters.push_back(nc);
        }
    }

    out_assembled.header.indexCount = uint32_t(out_assembled.indices.size());
    out_assembled.header.vertexCount =
        uint32_t(out_assembled.positions.size() / 3);
    out_assembled.header.levelCount = level_count;
    mAssembledDirty = false;
    return true;
}

void collectStreamingRequests(const TesseraData &data,
                              const float4x4 &viewProj,
                              float projScaleY,
                              float viewportHeight,
                              const TesseraLodParams &params,
                              std::vector<uint32_t> &out_requests)
{
    out_requests.clear();
    if (data.clusters.empty() || viewportHeight <= 0.0f) {
        return;
    }

    const TesseraViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);

    std::vector<uint32_t> children_count(data.clusters.size(), 0);
    for (const auto &c : data.clusters) {
        if (c.parentGroup >= 0 &&
            uint32_t(c.parentGroup) < data.clusters.size())
        {
            children_count[uint32_t(c.parentGroup)]++;
        }
    }

    const auto cluster_test = [&](const TesseraCluster &cluster) {
        if (cluster.parentError >= std::numeric_limits<float>::max()) {
            return false;
        }
        const float3 c(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        return projectedErrorPx(c, cluster.sphere[3], cluster.parentError,
                                view.camPos, viewportHeight, view.projScaleY,
                                view.orthographic) <= params.pixelError;
    };

    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const TesseraCluster &cluster = data.clusters[ci];
        const float3 center(cluster.sphere[0], cluster.sphere[1],
                            cluster.sphere[2]);
        const float radius = cluster.sphere[3];
        if (!sphere_in_frustum(view.planes, center, radius, params.cullingMargin)) {
            continue;
        }

        /* Reachability ignores residency: a non-resident ancestor never
         * emits, so it cannot cover this cluster. */
        bool ancestor_emitted = false;
        int32_t ancestor = cluster.parentGroup;
        while (ancestor >= 0 &&
               uint32_t(ancestor) < data.clusters.size())
        {
            const TesseraCluster &a = data.clusters[uint32_t(ancestor)];
            if (a.indexCount == 0) {
                ancestor = a.parentGroup;
                continue;
            }
            const float3 ac(a.sphere[0], a.sphere[1], a.sphere[2]);
            if (!sphere_in_frustum(view.planes, ac, a.sphere[3])) {
                ancestor_emitted = true;
                break;
            }
            if (cluster_test(a)) {
                ancestor_emitted = true;
                break;
            }
            ancestor = a.parentGroup;
        }
        if (ancestor_emitted) {
            continue;
        }

        const bool leaf = children_count[ci] == 0;
        if (!(leaf || cluster_test(cluster))) {
            continue;
        }
        if (cluster.indexCount != 0) {
            continue; /* resident and drawable */
        }
        out_requests.push_back(ci);
        /* Prefetch the missing ancestors as well: without them the LOD
         * cuts cannot engage (every visible leaf would be drawn) and the
         * resident set never settles. */
        int32_t up = cluster.parentGroup;
        while (up >= 0 && uint32_t(up) < data.clusters.size()) {
            const TesseraCluster &a = data.clusters[uint32_t(up)];
            if (a.indexCount == 0) {
                out_requests.push_back(uint32_t(up));
            }
            up = a.parentGroup;
        }
    }
}

}  // namespace tessera
}  // namespace blender
