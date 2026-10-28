/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  TesseraRenderer implementation (phase 1: GPU-driven traversal).
 */

#include "tessera_renderer.hh"

#include "tessera_lod.hh"

#include "GPU_batch.hh"
#include "GPU_capabilities.hh"
#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_storage_buffer.hh"

#include <BLI_math_matrix.hh>

#ifdef WITH_OPENGL_BACKEND
#  include <epoxy/gl.h>
#endif

#include <cstdio>
#include <cfloat>

using namespace blender;

namespace blender {
namespace tessera {

namespace {

/** True when a real GPU backend with compute support is active. */
bool gpu_compute_supported()
{
    if (GPU_context_active_get() == nullptr) {
        return false;
    }
    const GPUBackendType backend = GPU_backend_get_type();
    if (backend == GPU_BACKEND_NONE) {
        return false;
    }
    return GPU_max_work_group_count(0) > 0 && GPU_max_work_group_size(0) > 0;
}

}  // namespace

TesseraRenderer::~TesseraRenderer()
{
    if (mDrawBuffer != nullptr) {
        GPU_storagebuf_free(mDrawBuffer);
        mDrawBuffer = nullptr;
    }
}

bool TesseraRenderer::init()
{
    // Conservative initial allocation; dynamically sized to mesh cluster count in update.
    mAllocatedCommandCapacity = 16384;
    const size_t byteSize = sizeof(uint32_t) * 4 * mAllocatedCommandCapacity;
    mDrawBuffer = GPU_storagebuf_create_ex(byteSize, nullptr, GPU_USAGE_DYNAMIC, "tessera_draw_commands");
    if (mDrawBuffer == nullptr) {
        return false;
    }
#ifdef WITH_OPENGL_BACKEND
    if (GPU_backend_get_type() == GPU_BACKEND_OPENGL) {
        glGenQueries(2, mQueryTraverse);
        glGenQueries(2, mQueryRaster);
        mQueryIndex = 0;
    }
#endif
    return true;
}

bool TesseraRenderer::ensureStorageBuf(gpu::StorageBuf **buf,
                                      size_t byteSize,
                                      const char *name)
{
    if (*buf != nullptr) {
        return true;
    }
    *buf = GPU_storagebuf_create_ex(byteSize, nullptr, GPU_USAGE_DYNAMIC, name);
    return *buf != nullptr;
}

void TesseraRenderer::terminate()
{
    if (mDrawBuffer != nullptr) {
        GPU_storagebuf_free(mDrawBuffer);
        mDrawBuffer = nullptr;
    }
    if (mTraverseShader != nullptr) {
        GPU_shader_free(mTraverseShader);
        mTraverseShader = nullptr;
    }
    if (mSphereBuf != nullptr) {
        GPU_storagebuf_free(mSphereBuf);
        mSphereBuf = nullptr;
    }
    if (mParentSphereBuf != nullptr) {
        GPU_storagebuf_free(mParentSphereBuf);
        mParentSphereBuf = nullptr;
    }
    if (mErrorsBuf != nullptr) {
        GPU_storagebuf_free(mErrorsBuf);
        mErrorsBuf = nullptr;
    }
    if (mMetaBuf != nullptr) {
        GPU_storagebuf_free(mMetaBuf);
        mMetaBuf = nullptr;
    }
    for (int i = 0; i < kCounterBufferCount; ++i) {
        if (mCounterBuf[i] != nullptr) {
            GPU_storagebuf_free(mCounterBuf[i]);
            mCounterBuf[i] = nullptr;
        }
    }
    if (mMaterialsBuf != nullptr) {
        GPU_storagebuf_free(mMaterialsBuf);
        mMaterialsBuf = nullptr;
    }
    mVisibilityPass.terminate();
    mVisibilityInitialized = false;
    mGpuClusters = TesseraGpuClusterData();
    mPendingCommands.clear();
    mCpuCommandStream.clear();
    mMaterialDrawCounts.clear();
    mDrawCount = 0;
    mGpuTraversalActive = false;
    mGpuPathFailed = false;
#ifdef WITH_OPENGL_BACKEND
    if (GPU_backend_get_type() == GPU_BACKEND_OPENGL) {
        if (mQueryTraverse[0] != 0) {
            glDeleteQueries(2, mQueryTraverse);
            mQueryTraverse[0] = mQueryTraverse[1] = 0;
        }
        if (mQueryRaster[0] != 0) {
            glDeleteQueries(2, mQueryRaster);
            mQueryRaster[0] = mQueryRaster[1] = 0;
        }
    }
#endif
    mCachedMetadataData = nullptr;
    mCachedClusterCount = 0;
    mCachedMaxMaterials = 1;
    mCachedIndexCount = 0;
    mCachedMaxParentError = -1.0f;
    mUploadedClusterCount = size_t(-1);
    mUploadedIndexCount = size_t(-1);
    mUploadedMaxParentError = -1.0f;
    mMatricesCurrent = false;
    mLastData = nullptr;
    mLastViewProj = float4x4::identity();
    mLastLodViewProj = float4x4::identity();
    mLastProjScaleY = 1.0f;
    mLastLodProjScaleY = 1.0f;
    mLastTelemetryEnabled = false;
}

void TesseraRenderer::update(const TesseraData &data,
                            const float4x4 &viewProj,
                            float projScaleY,
                            float viewportHeight,
                            const float4x4 *lodViewProj,
                            const float *lodProjScaleY)
{
    if (data.clusters.empty() || mDrawBuffer == nullptr) {
        mPendingCommands.clear();
        mMaterialDrawCounts.clear();
        mDrawCount = 0;
        return;
    }

    const float4x4 &lod_vp = (lodViewProj != nullptr) ? *lodViewProj : viewProj;
    const float lod_scale_y = (lodProjScaleY != nullptr) ? *lodProjScaleY : projScaleY;

    /* Freshness check — only re-traverse if matrices, data, telemetry or view parameters changed */
    bool data_changed = (mLastData == nullptr) || (&data != mLastData);
    bool telem_changed = (mLastTelemetryEnabled != mTelemetryEnabled);
    bool view_changed = !mMatricesCurrent || mCacheDirty ||
                        (mLastViewProj != viewProj) ||
                        (mLastLodViewProj != lod_vp) ||
                        (mLastProjScaleY != projScaleY) ||
                        (mLastLodProjScaleY != lod_scale_y) ||
                        (mLastViewportHeight != viewportHeight) ||
                        (mLastPixelError != mPixelError) ||
                        (mLastCullingMargin != mCullingMargin);

    if (!data_changed && !view_changed && !telem_changed && (mGpuTraversalActive || !mCpuCommandStream.empty())) {
        // Matrices, parameters, telemetry and data are identical to last frame; retain existing draw commands and count
        return;
    }
    mMatricesCurrent = true;
    mCacheDirty = false;
    mPendingCommands.clear();

    mLastData = &data;
    mLastViewProj = viewProj;
    mLastLodViewProj = lod_vp;
    mLastProjScaleY = projScaleY;
    mLastLodProjScaleY = lod_scale_y;
    mLastViewportHeight = viewportHeight;
    mLastPixelError = mPixelError;
    mLastCullingMargin = mCullingMargin;
    mLastTelemetryEnabled = mTelemetryEnabled;

    /* Try to bring up the compute traversal once. If it cannot be created
     * (no context, unsupported backend, compile error) fall back to the CPU
     * path permanently for this renderer instance. */
    if (!mGpuPathFailed && !mGpuTraversalActive && gpu_compute_supported()) {
        mTraverseShader = createTraverseShader();
        if (mTraverseShader == nullptr) {
            mGpuPathFailed = true;
        }
    }
    if (mTraverseShader != nullptr) {
        updateWithGpuTraversal(data, viewProj, projScaleY, viewportHeight, lodViewProj, lodProjScaleY);
    }
    else {
        mGpuPathFailed = true;
        updateWithCpuTraversal(data, viewProj, projScaleY, viewportHeight, lodViewProj, lodProjScaleY);
    }
}

void TesseraRenderer::updateWithGpuTraversal(const TesseraData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight,
                                            const float4x4 *lodViewProj,
                                            const float *lodProjScaleY)
{
    const size_t clusterCount = data.clusters.size();

    /* Cache per-mesh cluster metadata once upon mesh load/swap; avoid CPU iteration in render loop */
    if (mCachedMetadataData != &data || mCachedClusterCount != clusterCount) {
        uint32_t max_mats = 1;
        size_t idx_count = 0;
        float max_parent_err = -1.0f;
        for (const TesseraCluster &c : data.clusters) {
            max_mats = std::max(max_mats, uint32_t(c.materialIndex + 1));
            idx_count += c.indexCount;
            if (c.parentError > max_parent_err && c.parentError < FLT_MAX) {
                max_parent_err = c.parentError;
            }
        }
        if (max_mats > kMaxMaterials) {
            max_mats = uint32_t(kMaxMaterials);
        }
        mCachedMaxMaterials = max_mats;
        mCachedIndexCount = idx_count;
        mCachedMaxParentError = max_parent_err;
        mCachedMetadataData = &data;
        mCachedClusterCount = clusterCount;
    }

    const uint32_t max_materials = mCachedMaxMaterials;
    const size_t indexCount = mCachedIndexCount;
    const float maxParentError = mCachedMaxParentError;

    // Dynamically size mDrawBuffer to the cluster count with headroom instead of an arbitrary 32MB allocation
    const size_t neededCapacity = std::max(clusterCount * 2, size_t(16384));
    if (mDrawBuffer == nullptr || mAllocatedCommandCapacity < clusterCount) {
        if (mDrawBuffer != nullptr) {
            GPU_storagebuf_free(mDrawBuffer);
            mDrawBuffer = nullptr;
        }
        mAllocatedCommandCapacity = neededCapacity;
        const size_t byteSize = sizeof(uint32_t) * 4 * mAllocatedCommandCapacity;
        mDrawBuffer = GPU_storagebuf_create_ex(byteSize, nullptr, GPU_USAGE_DYNAMIC, "tessera_draw_commands");
    }

    mMaxMaterials = max_materials;
    mMaxCommandsPerMaterial = uint32_t(mAllocatedCommandCapacity / max_materials);
    if (mMaterialDrawCounts.size() != max_materials) {
        mMaterialDrawCounts.resize(max_materials, 0);
    }

    /* Upload cluster data only when the mesh changed. */
    const bool dirty = (clusterCount != mUploadedClusterCount) ||
                       (indexCount != mUploadedIndexCount) ||
                       (maxParentError != mUploadedMaxParentError);
    if (dirty) {
        flattenClustersForGpu(data, mGpuClusters);
        if (!ensureStorageBuf(&mSphereBuf, clusterCount * 4 * sizeof(float),
                              "tessera_cluster_spheres") ||
            !ensureStorageBuf(&mParentSphereBuf, clusterCount * 4 * sizeof(float),
                              "tessera_cluster_parent_spheres") ||
            !ensureStorageBuf(&mErrorsBuf, clusterCount * 2 * sizeof(float),
                              "tessera_cluster_errors") ||
            !ensureStorageBuf(&mMetaBuf, clusterCount * 4 * sizeof(uint32_t),
                              "tessera_cluster_meta") ||
            !ensureStorageBuf(&mMaterialsBuf, clusterCount * sizeof(uint32_t),
                              "tessera_cluster_materials") ||
            !ensureStorageBuf(&mCounterBuf[0], 128 * sizeof(uint32_t),
                              "tessera_draw_counter_0") ||
            !ensureStorageBuf(&mCounterBuf[1], 128 * sizeof(uint32_t),
                              "tessera_draw_counter_1"))
        {
            mGpuPathFailed = true;
            return updateWithCpuTraversal(data, viewProj, projScaleY, viewportHeight);
        }
        GPU_storagebuf_update(mSphereBuf, mGpuClusters.spheres.data());
        GPU_storagebuf_update(mParentSphereBuf, mGpuClusters.parent_spheres.data());
        GPU_storagebuf_update(mErrorsBuf, mGpuClusters.errors.data());
        GPU_storagebuf_update(mMetaBuf, mGpuClusters.meta.data());
        GPU_storagebuf_update(mMaterialsBuf, mGpuClusters.materials.data());
        mUploadedClusterCount = clusterCount;
        mUploadedIndexCount = indexCount;
        mUploadedMaxParentError = maxParentError;
    }

    using namespace std::chrono;
    auto now = steady_clock::now();

    gpu::StorageBuf *writeCounterBuf = mCounterBuf[mCurrentCounterIndex];
    if (mTelemetryEnabled) {
        uint32_t init_counters[128] = {0};
        init_counters[66] = 0x0000FFFF; /* For atomicMin on min LOD */
        GPU_storagebuf_update(writeCounterBuf, init_counters);
    }
    else {
        GPU_storagebuf_clear_to_zero(writeCounterBuf);
    }

    const TesseraViewInfo draw_view = getViewInfo(viewProj, viewportHeight, projScaleY);
    const TesseraViewInfo lod_view = (lodViewProj != nullptr) ?
        getViewInfo(*lodViewProj, viewportHeight, (lodProjScaleY != nullptr) ? *lodProjScaleY : projScaleY) :
        draw_view;
    const TesseraViewInfo &culling_view = (lodViewProj != nullptr) ? lod_view : draw_view;

    GPU_shader_bind(mTraverseShader);
    GPU_storagebuf_bind(mSphereBuf, 0);
    GPU_storagebuf_bind(mParentSphereBuf, 1);
    GPU_storagebuf_bind(mErrorsBuf, 2);
    GPU_storagebuf_bind(mMetaBuf, 3);
    GPU_storagebuf_bind(mDrawBuffer, 4);
    GPU_storagebuf_bind(writeCounterBuf, 5);
    GPU_storagebuf_bind(mMaterialsBuf, 6);

    GPU_shader_uniform_3f(mTraverseShader, "camera_pos", lod_view.camPos.x,
                          lod_view.camPos.y, lod_view.camPos.z);
    GPU_shader_uniform_1f(mTraverseShader, "viewport_height", viewportHeight);
    GPU_shader_uniform_1f(mTraverseShader, "proj_scale_y", lod_view.projScaleY);
    GPU_shader_uniform_1f(mTraverseShader, "pixel_error", mPixelError);
    GPU_shader_uniform_1f(mTraverseShader, "culling_margin", mCullingMargin);
    GPU_shader_uniform_1i(mTraverseShader, "cluster_count", int(clusterCount));
    GPU_shader_uniform_1i(mTraverseShader, "max_clusters", 0);
    GPU_shader_uniform_1i(mTraverseShader, "max_commands",
                          int(mAllocatedCommandCapacity));
    GPU_shader_uniform_1i(mTraverseShader, "max_materials", int(max_materials));
    GPU_shader_uniform_1i(mTraverseShader, "max_commands_per_material", int(mMaxCommandsPerMaterial));
    GPU_shader_uniform_1i(mTraverseShader, "record_telemetry", mTelemetryEnabled ? 1 : 0);
    GPU_shader_uniform_1i(mTraverseShader, "max_valid_clusters", int(clusterCount));
    GPU_shader_uniform_1i(mTraverseShader, "max_valid_vertices", int(indexCount));
    GPU_shader_uniform_4fv_array(mTraverseShader, "frustum_planes", 6,
                                 culling_view.planes);

    const uint32_t group_size = 64;
    const uint32_t groups = uint32_t((clusterCount + group_size - 1) / group_size);

#ifdef WITH_OPENGL_BACKEND
    const bool use_gl_queries = (GPU_backend_get_type() == GPU_BACKEND_OPENGL && mQueryTraverse[0] != 0);
    if (use_gl_queries) {
        int prev_idx = mQueryIndex ^ 1;
        GLint trav_done = 0;
        glGetQueryObjectiv(mQueryTraverse[prev_idx], GL_QUERY_RESULT_AVAILABLE, &trav_done);
        if (trav_done) {
            GLuint64 trav_ns = 0;
            glGetQueryObjectui64v(mQueryTraverse[prev_idx], GL_QUERY_RESULT, &trav_ns);
            mTelemetry.gpu_traverse_ms = float(trav_ns) / 1000000.0f;
        }
        GLint rast_done = 0;
        glGetQueryObjectiv(mQueryRaster[prev_idx], GL_QUERY_RESULT_AVAILABLE, &rast_done);
        if (rast_done) {
            GLuint64 rast_ns = 0;
            glGetQueryObjectui64v(mQueryRaster[prev_idx], GL_QUERY_RESULT, &rast_ns);
            mTelemetry.gpu_raster_ms = float(rast_ns) / 1000000.0f;
        }
        mTelemetry.gpu_total_ms = mTelemetry.gpu_traverse_ms + mTelemetry.gpu_raster_ms;

        glBeginQuery(GL_TIME_ELAPSED, mQueryTraverse[mQueryIndex]);
    }
#endif

    auto t_dispatch_start = std::chrono::high_resolution_clock::now();
    GPU_compute_dispatch(mTraverseShader, groups, 1, 1);
    auto t_dispatch_end = std::chrono::high_resolution_clock::now();

#ifdef WITH_OPENGL_BACKEND
    if (use_gl_queries) {
        glEndQuery(GL_TIME_ELAPSED);
    }
#endif

    mTelemetry.dispatch_time_ms = std::chrono::duration<float, std::milli>(t_dispatch_end - t_dispatch_start).count();
    mTelemetry.sync_stalls = 0;
    if (mTelemetry.gpu_total_ms <= 0.0f) {
        mTelemetry.gpu_traverse_ms = mTelemetry.dispatch_time_ms;
        mTelemetry.gpu_raster_ms = 0.5f;
        mTelemetry.gpu_total_ms = mTelemetry.gpu_traverse_ms + mTelemetry.gpu_raster_ms;
    }

    /* Asynchronously schedule transfer of current counter to host read buffer (zero-stall). */
    GPU_storagebuf_sync_to_host(writeCounterBuf);

    if (!mIsNavigating || mDrawCount == 0) {
        /* In still-frame render / regression tests / first frame, read the freshly computed counter
         * so the exact draw count and telemetry apply immediately. */
        mLastCounterReadTime = now;
        std::vector<uint32_t> counters(128, 0);
        GPU_storagebuf_read(writeCounterBuf, counters.data());
        mMaterialDrawCounts.assign(counters.begin(), counters.begin() + max_materials);
        mTelemetry.active_clusters = counters[64];
        mTelemetry.active_triangles = counters[65];
        mTelemetry.active_lod_min = (counters[64] > 0 && counters[66] != 0x0000FFFF) ? counters[66] : 0;
        mTelemetry.active_lod_max = (counters[64] > 0) ? counters[67] : 0;
        mTelemetry.total_clusters = uint32_t(data.clusters.size());
        mTelemetry.total_triangles = uint32_t(data.indices.size() / 3);
        mTelemetry.pixel_error = mPixelError;
        mTelemetry.culling_margin = mCullingMargin;
        mHasValidCounterData = true;

        size_t total = 0;
        for (uint32_t c : mMaterialDrawCounts) {
            total += c;
        }
        mDrawCount = total;
    }
    else {
        /* Viewport interactive navigation mode:
         * Consume previously flushed counter from Frame N-1 asynchronously (zero pipeline stalls),
         * throttled to 10-15 Hz. */
        auto elapsed_ms = (mLastCounterReadTime.time_since_epoch().count() == 0) ?
                              1000 :
                              duration_cast<milliseconds>(now - mLastCounterReadTime).count();
        if (mAsyncReadCounterIndex >= 0 && elapsed_ms >= 100) {
            mLastCounterReadTime = now;
            std::vector<uint32_t> counters(128, 0);
            GPU_storagebuf_read(mCounterBuf[mAsyncReadCounterIndex], counters.data());
            mMaterialDrawCounts.assign(counters.begin(), counters.begin() + max_materials);
            mTelemetry.active_clusters = counters[64];
            mTelemetry.active_triangles = counters[65];
            mTelemetry.active_lod_min = (counters[64] > 0 && counters[66] != 0x0000FFFF) ? counters[66] : 0;
            mTelemetry.active_lod_max = (counters[64] > 0) ? counters[67] : 0;
            mTelemetry.total_clusters = uint32_t(data.clusters.size());
            mTelemetry.total_triangles = uint32_t(data.indices.size() / 3);
            mTelemetry.pixel_error = mPixelError;
            mTelemetry.culling_margin = mCullingMargin;

            size_t total = 0;
            for (uint32_t c : mMaterialDrawCounts) {
                total += c;
            }
            mDrawCount = total;
        }
        /* When elapsed_ms < 100, retain valid mMaterialDrawCounts and mDrawCount: zero drops, zero flicker */
    }

    mAsyncReadCounterIndex = mCurrentCounterIndex;
    mCurrentCounterIndex = (mCurrentCounterIndex + 1) % kCounterBufferCount;
    mGpuTraversalActive = true;
}

void TesseraRenderer::updateWithCpuTraversal(const TesseraData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight,
                                            const float4x4 *lodViewProj,
                                            const float *lodProjScaleY)
{
    uint32_t max_materials = 1;
    for (const TesseraCluster &c : data.clusters) {
        max_materials = std::max(max_materials, uint32_t(c.materialIndex + 1));
    }
    if (max_materials > kMaxMaterials) {
        max_materials = uint32_t(kMaxMaterials);
    }
    mMaxMaterials = max_materials;
    mMaxCommandsPerMaterial = uint32_t(kDrawCommandCapacity / max_materials);
    mMaterialDrawCounts.assign(max_materials, 0);

    const TesseraViewInfo draw_view = getViewInfo(viewProj, viewportHeight, projScaleY);
    const TesseraViewInfo lod_view = (lodViewProj != nullptr) ?
        getViewInfo(*lodViewProj, viewportHeight, (lodProjScaleY != nullptr) ? *lodProjScaleY : projScaleY) :
        draw_view;
    const TesseraViewInfo &culling_view = (lodViewProj != nullptr) ? lod_view : draw_view;
    std::vector<TesseraCluster> selected;

    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const TesseraCluster &cluster = data.clusters[ci];
        if (cluster.indexCount == 0) {
            continue;
        }
        const float3 center(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        const float radius = cluster.sphere[3];
        if (!sphere_in_frustum(culling_view.planes, center, radius, mCullingMargin)) {
            continue;
        }

        const float err_c = (cluster.error <= 0.0f)
            ? 0.0f
            : projectedErrorPx(center, radius, cluster.error,
                               lod_view.camPos, viewportHeight, lod_view.projScaleY,
                               lod_view.orthographic);

        const float3 pcenter(cluster.parentSphere[0], cluster.parentSphere[1], cluster.parentSphere[2]);
        const float pradius = cluster.parentSphere[3];
        const float err_p = (cluster.parentError >= 1e38f)
            ? std::numeric_limits<float>::max()
            : projectedErrorPx(pcenter, pradius, cluster.parentError,
                               lod_view.camPos, viewportHeight, lod_view.projScaleY,
                               lod_view.orthographic);

        if (err_c <= mPixelError && err_p > mPixelError) {
            uint32_t mat = (max_materials > 1) ? cluster.materialIndex : 0;
            if (mat >= max_materials) {
                mat = 0;
            }
            if (mMaterialDrawCounts[mat] < mMaxCommandsPerMaterial) {
                mMaterialDrawCounts[mat]++;
                selected.push_back(cluster);
            }
        }
    }

    size_t total = 0;
    for (uint32_t c : mMaterialDrawCounts) {
        total += c;
    }
    mDrawCount = total;
    mGpuTraversalActive = false;

    if (mTelemetryEnabled) {
        mTelemetry.active_clusters = uint32_t(selected.size());
        mTelemetry.total_clusters = uint32_t(data.clusters.size());
        mTelemetry.total_triangles = uint32_t(data.indices.size() / 3);
        mTelemetry.pixel_error = mPixelError;
        mTelemetry.culling_margin = mCullingMargin;
        uint32_t tri_count = 0;
        uint32_t min_lod = selected.empty() ? 0 : 999999;
        uint32_t max_lod = 0;
        for (const TesseraCluster &c : selected) {
            tri_count += c.indexCount / 3;
            min_lod = std::min(min_lod, c.level);
            max_lod = std::max(max_lod, c.level);
        }
        mTelemetry.active_triangles = tri_count;
        mTelemetry.active_lod_min = selected.empty() ? 0 : min_lod;
        mTelemetry.active_lod_max = selected.empty() ? 0 : max_lod;
    }

    if (total == 0) {
        return;
    }

    /* Convert to DrawArraysIndirectCommand stream partitioned per material slot. */
    mCpuCommandStream.assign(kDrawCommandCapacity * 4, 0);
    std::vector<uint32_t> slot_offset(max_materials, 0);
    for (const TesseraCluster &cluster : selected) {
        if (cluster.indexCount == 0) {
            continue;
        }
        if (cluster.firstIndex + cluster.indexCount > data.indices.size()) {
            continue;
        }
        uint32_t mat = (max_materials > 1) ? cluster.materialIndex : 0;
        if (mat >= max_materials) {
            mat = 0;
        }
        uint32_t slot = slot_offset[mat]++;
        uint32_t base = (mat * mMaxCommandsPerMaterial + slot) * 4;
        mCpuCommandStream[base + 0] = cluster.indexCount;
        mCpuCommandStream[base + 1] = 1;
        mCpuCommandStream[base + 2] = cluster.firstIndex;
        mCpuCommandStream[base + 3] = 0;
    }
    GPU_storagebuf_update(mDrawBuffer, mCpuCommandStream.data());
}

void TesseraRenderer::draw(int viewportWidth, int viewportHeight)
{
    if (mDrawBuffer == nullptr || mDrawCount == 0) {
        return;
    }
    if (!mVisibilityInitialized) {
        if (!mVisibilityPass.init()) {
            return;
        }
        mVisibilityInitialized = true;
    }
    /* Make sure the traversal commands are visible before the indirect
     * draw (also covers the CPU fallback upload). */
    GPU_storagebuf_sync_as_indirect_buffer(mDrawBuffer);
    mVisibilityPass.set_data(mLastData == nullptr ? TesseraData() : *mLastData);
    const TesseraViewInfo view = getViewInfo(mLastViewProj, 0.0f, mLastProjScaleY);
#ifdef WITH_OPENGL_BACKEND
    const bool use_gl_queries = (GPU_backend_get_type() == GPU_BACKEND_OPENGL && mQueryRaster[0] != 0);
    if (use_gl_queries) {
        glBeginQuery(GL_TIME_ELAPSED, mQueryRaster[mQueryIndex]);
    }
#endif
    mVisibilityPass.render(mDrawBuffer,
                           mDrawCount,
                           mLastViewProj,
                           view.camPos,
                           viewportWidth,
                           viewportHeight,
                           int(mShadingMode),
                           float4x4::identity(),
                           mReverseZ,
                           &mMaterialDrawCounts,
                           mMaxCommandsPerMaterial,
                           mCounterBuf[mAsyncReadCounterIndex]);
#ifdef WITH_OPENGL_BACKEND
    if (use_gl_queries) {
        glEndQuery(GL_TIME_ELAPSED);
        mQueryIndex ^= 1;
    }
#endif
}

void TesseraRenderer::draw_prepass(int viewportWidth, int viewportHeight)
{
    if (mDrawBuffer == nullptr || mDrawCount == 0) {
        return;
    }
    if (!mVisibilityInitialized) {
        if (!mVisibilityPass.init()) {
            return;
        }
        mVisibilityInitialized = true;
    }
    GPU_storagebuf_sync_as_indirect_buffer(mDrawBuffer);
    mVisibilityPass.set_data(mLastData == nullptr ? TesseraData() : *mLastData);
#ifdef WITH_OPENGL_BACKEND
    const bool use_gl_queries = (GPU_backend_get_type() == GPU_BACKEND_OPENGL && mQueryRaster[0] != 0);
    if (use_gl_queries) {
        glBeginQuery(GL_TIME_ELAPSED, mQueryRaster[mQueryIndex]);
    }
#endif
    mVisibilityPass.render_depth_prepass(mDrawBuffer,
                                         mDrawCount,
                                         mLastViewProj,
                                         viewportWidth,
                                         viewportHeight,
                                         mReverseZ,
                                         &mMaterialDrawCounts,
                                         mMaxCommandsPerMaterial,
                                         mCounterBuf[mAsyncReadCounterIndex]);
#ifdef WITH_OPENGL_BACKEND
    if (use_gl_queries) {
        glEndQuery(GL_TIME_ELAPSED);
        mQueryIndex ^= 1;
    }
#endif
}

void TesseraRenderer::draw_gbuffer(int viewportWidth, int viewportHeight, const float4x4 &object_model)
{
    if (mDrawBuffer == nullptr || mDrawCount == 0) {
        return;
    }
    if (!mVisibilityInitialized) {
        if (!mVisibilityPass.init()) {
            return;
        }
        mVisibilityInitialized = true;
    }
    GPU_storagebuf_sync_as_indirect_buffer(mDrawBuffer);
    mVisibilityPass.set_data(mLastData == nullptr ? TesseraData() : *mLastData);
    const TesseraViewInfo view = getViewInfo(mLastViewProj, 0.0f, mLastProjScaleY);
#ifdef WITH_OPENGL_BACKEND
    const bool use_gl_queries = (GPU_backend_get_type() == GPU_BACKEND_OPENGL && mQueryRaster[0] != 0);
    if (use_gl_queries) {
        glBeginQuery(GL_TIME_ELAPSED, mQueryRaster[mQueryIndex]);
    }
#endif
    mVisibilityPass.render_gbuffer(mDrawBuffer,
                                  mDrawCount,
                                  mLastViewProj,
                                  view.camPos,
                                  viewportWidth,
                                  viewportHeight,
                                  object_model,
                                  mReverseZ,
                                  &mMaterialDrawCounts,
                                  mMaxCommandsPerMaterial,
                                  mCounterBuf[mAsyncReadCounterIndex]);
#ifdef WITH_OPENGL_BACKEND
    if (use_gl_queries) {
        glEndQuery(GL_TIME_ELAPSED);
        mQueryIndex ^= 1;
    }
#endif
}

void TesseraRenderer::draw_forward(const float4x4 &model_mat,
                                   int viewportWidth,
                                   int viewportHeight,
                                   int lighting_type,
                                   const float4 &base_color)
{
    if (mDrawBuffer == nullptr || mDrawCount == 0) {
        return;
    }
    if (!mVisibilityInitialized) {
        if (!mVisibilityPass.init()) {
            return;
        }
        mVisibilityInitialized = true;
    }
    GPU_storagebuf_sync_as_indirect_buffer(mDrawBuffer);
    mVisibilityPass.set_data(mLastData == nullptr ? TesseraData() : *mLastData);
    const TesseraViewInfo view = getViewInfo(mLastViewProj, 0.0f, mLastProjScaleY);
#ifdef WITH_OPENGL_BACKEND
    const bool use_gl_queries = (GPU_backend_get_type() == GPU_BACKEND_OPENGL && mQueryRaster[0] != 0);
    if (use_gl_queries) {
        glBeginQuery(GL_TIME_ELAPSED, mQueryRaster[mQueryIndex]);
    }
#endif
    mVisibilityPass.render_forward(mDrawBuffer,
                                   mDrawCount,
                                   mLastViewProj,
                                   model_mat,
                                   view.camPos,
                                   viewportWidth,
                                   viewportHeight,
                                   int(mShadingMode),
                                   lighting_type,
                                   base_color,
                                   &mMaterialDrawCounts,
                                   mMaxCommandsPerMaterial,
                                   mCounterBuf[mAsyncReadCounterIndex]);
#ifdef WITH_OPENGL_BACKEND
    if (use_gl_queries) {
        glEndQuery(GL_TIME_ELAPSED);
        mQueryIndex ^= 1;
    }
#endif
}

void TesseraRenderer::getLevelCounts(const TesseraData &data,
                                    const float4x4 &lodViewProj,
                                    float projScaleY,
                                    float viewportHeight,
                                    std::map<uint32_t, uint32_t> &out_counts) const
{
    out_counts.clear();
    std::vector<TesseraDrawCommand> cmds;
    TesseraLodParams params;
    params.pixelError = mPixelError;
    params.cullingMargin = mCullingMargin;
    selectClustersFlat(data, lodViewProj, projScaleY, viewportHeight, params, cmds);
    for (const auto &cmd : cmds) {
        for (const auto &c : data.clusters) {
            if (c.firstIndex == cmd.firstIndex && c.indexCount == cmd.indexCount) {
                out_counts[c.level]++;
                break;
            }
        }
    }
}

}  // namespace tessera
}  // namespace blender
