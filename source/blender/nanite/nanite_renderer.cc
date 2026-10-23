/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  NaniteRenderer implementation (phase 1: GPU-driven traversal).
 */

#include "nanite_renderer.hh"

#include "nanite_lod.hh"

#include "GPU_batch.hh"
#include "GPU_capabilities.hh"
#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_storage_buffer.hh"

#include <cstdio>
#include <cfloat>

using namespace blender;

namespace blender {
namespace nanite {

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

NaniteRenderer::~NaniteRenderer()
{
    if (mDrawBuffer != nullptr) {
        GPU_storagebuf_free(mDrawBuffer);
        mDrawBuffer = nullptr;
    }
}

bool NaniteRenderer::init()
{
    // Reserve a large static capacity for indirect draw commands.
    // Each command is a DrawArraysIndirectCommand (4 x uint32).
    const size_t byteSize = sizeof(uint32_t) * 4 * kDrawCommandCapacity;
    mDrawBuffer = GPU_storagebuf_create(byteSize);
    if (mDrawBuffer == nullptr) {
        return false;
    }
    return true;
}

bool NaniteRenderer::ensureStorageBuf(gpu::StorageBuf **buf,
                                      size_t byteSize,
                                      const char *name)
{
    if (*buf != nullptr) {
        return true;
    }
    *buf = GPU_storagebuf_create_ex(byteSize, nullptr, GPU_USAGE_DYNAMIC, name);
    return *buf != nullptr;
}

void NaniteRenderer::terminate()
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
    if (mCounterBuf != nullptr) {
        GPU_storagebuf_free(mCounterBuf);
        mCounterBuf = nullptr;
    }
    mVisibilityPass.terminate();
    mVisibilityInitialized = false;
    mGpuClusters = NaniteGpuClusterData();
    mPendingCommands.clear();
    mCpuCommandStream.clear();
    mDrawCount = 0;
    mGpuTraversalActive = false;
    mGpuPathFailed = false;
    mUploadedClusterCount = size_t(-1);
    mUploadedIndexCount = size_t(-1);
    mUploadedMaxParentError = -1.0f;
    mMatricesCurrent = false;
    mLastData = nullptr;
    mLastViewProj = float4x4::identity();
    mLastLodViewProj = float4x4::identity();
    mLastProjScaleY = 1.0f;
    mLastLodProjScaleY = 1.0f;
}

void NaniteRenderer::update(const NaniteData &data,
                            const float4x4 &viewProj,
                            float projScaleY,
                            float viewportHeight,
                            const float4x4 *lodViewProj,
                            const float *lodProjScaleY)
{
    if (data.clusters.empty() || mDrawBuffer == nullptr) {
        mPendingCommands.clear();
        mDrawCount = 0;
        return;
    }

    const float4x4 &lod_vp = (lodViewProj != nullptr) ? *lodViewProj : viewProj;
    const float lod_scale_y = (lodProjScaleY != nullptr) ? *lodProjScaleY : projScaleY;

    /* Freshness check — only re-traverse if matrices, data or view parameters changed */
    bool data_changed = (mLastData == nullptr) || (&data != mLastData);
    bool view_changed = !mMatricesCurrent ||
                        (mLastViewProj != viewProj) ||
                        (mLastLodViewProj != lod_vp) ||
                        (mLastProjScaleY != projScaleY) ||
                        (mLastLodProjScaleY != lod_scale_y) ||
                        (mLastViewportHeight != viewportHeight) ||
                        (mLastPixelError != mPixelError);

    if (!data_changed && !view_changed && (mGpuTraversalActive || !mCpuCommandStream.empty())) {
        // Matrices, parameters and data are identical to last frame; retain existing draw commands and count
        return;
    }
    mMatricesCurrent = true;
    mPendingCommands.clear();

    mLastData = &data;
    mLastViewProj = viewProj;
    mLastLodViewProj = lod_vp;
    mLastProjScaleY = projScaleY;
    mLastLodProjScaleY = lod_scale_y;
    mLastViewportHeight = viewportHeight;
    mLastPixelError = mPixelError;

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
        updateWithGpuTraversal(data, lod_vp, lod_scale_y, viewportHeight);
    }
    else {
        mGpuPathFailed = true;
        updateWithCpuTraversal(data, lod_vp, lod_scale_y, viewportHeight);
    }
}

void NaniteRenderer::updateWithGpuTraversal(const NaniteData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight)
{
    const size_t clusterCount = data.clusters.size();

    /* Upload cluster data only when the mesh changed. */
    size_t indexCount = 0;
    float maxParentError = -1.0f;
    for (const NaniteCluster &c : data.clusters) {
        indexCount += c.indexCount;
        maxParentError = (c.parentError > maxParentError && c.parentError < FLT_MAX)
                             ? c.parentError
                             : maxParentError;
    }
    const bool dirty = (clusterCount != mUploadedClusterCount) ||
                       (indexCount != mUploadedIndexCount) ||
                       (maxParentError != mUploadedMaxParentError);
    if (dirty) {
        flattenClustersForGpu(data, mGpuClusters);
        if (!ensureStorageBuf(&mSphereBuf, clusterCount * 4 * sizeof(float),
                              "nanite_cluster_spheres") ||
            !ensureStorageBuf(&mParentSphereBuf, clusterCount * 4 * sizeof(float),
                              "nanite_cluster_parent_spheres") ||
            !ensureStorageBuf(&mErrorsBuf, clusterCount * 2 * sizeof(float),
                              "nanite_cluster_errors") ||
            !ensureStorageBuf(&mMetaBuf, clusterCount * 4 * sizeof(uint32_t),
                              "nanite_cluster_meta") ||
            !ensureStorageBuf(&mCounterBuf, sizeof(uint32_t), "nanite_draw_counter"))
        {
            mGpuPathFailed = true;
            return updateWithCpuTraversal(data, viewProj, projScaleY, viewportHeight);
        }
        GPU_storagebuf_update(mSphereBuf, mGpuClusters.spheres.data());
        GPU_storagebuf_update(mParentSphereBuf, mGpuClusters.parent_spheres.data());
        GPU_storagebuf_update(mErrorsBuf, mGpuClusters.errors.data());
        GPU_storagebuf_update(mMetaBuf, mGpuClusters.meta.data());
        mUploadedClusterCount = clusterCount;
        mUploadedIndexCount = indexCount;
        mUploadedMaxParentError = maxParentError;
    }

    /* Reset the shared counter, bind resources and dispatch one thread per
     * cluster. */
    GPU_storagebuf_clear_to_zero(mCounterBuf);

    const NaniteViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);

    GPU_shader_bind(mTraverseShader);
    GPU_storagebuf_bind(mSphereBuf, 0);
    GPU_storagebuf_bind(mParentSphereBuf, 1);
    GPU_storagebuf_bind(mErrorsBuf, 2);
    GPU_storagebuf_bind(mMetaBuf, 3);
    GPU_storagebuf_bind(mDrawBuffer, 4);
    GPU_storagebuf_bind(mCounterBuf, 5);

    GPU_shader_uniform_3f(mTraverseShader, "camera_pos", view.camPos.x,
                          view.camPos.y, view.camPos.z);
    GPU_shader_uniform_1f(mTraverseShader, "viewport_height", viewportHeight);
    GPU_shader_uniform_1f(mTraverseShader, "proj_scale_y", view.projScaleY);
    GPU_shader_uniform_1f(mTraverseShader, "pixel_error", mPixelError);
    GPU_shader_uniform_1i(mTraverseShader, "cluster_count", int(clusterCount));
    GPU_shader_uniform_1i(mTraverseShader, "max_clusters", 0);
    GPU_shader_uniform_1i(mTraverseShader, "max_commands",
                          int(kDrawCommandCapacity));
    GPU_shader_uniform_4fv_array(mTraverseShader, "frustum_planes", 6,
                                 view.planes);

    const uint32_t group_size = 64;
    const uint32_t groups = uint32_t((clusterCount + group_size - 1) / group_size);
    GPU_compute_dispatch(mTraverseShader, groups, 1, 1);

    /* Read back the number of emitted commands (small, synchronized with the
     * dispatch). This is the only per-frame CPU<->GPU sync left. */
    uint32_t emitted = 0;
    GPU_storagebuf_read(mCounterBuf, &emitted);
    mDrawCount = size_t(emitted);
    mGpuTraversalActive = true;
}

void NaniteRenderer::updateWithCpuTraversal(const NaniteData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight)
{
    NaniteLodParams params;
    params.pixelError = mPixelError;
    const size_t count = selectClustersFlat(
        data, viewProj, projScaleY, viewportHeight, params, mPendingCommands);
    mDrawCount = count;
    mGpuTraversalActive = false;

    if (count == 0) {
        return;
    }

    /* Convert to DrawArraysIndirectCommand stream (16 bytes per command):
     * the visibility pass uses vertex pulling, so `first` is the cluster's
     * absolute index-buffer position. */
    mCpuCommandStream.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
        mCpuCommandStream[i * 4 + 0] = mPendingCommands[i].indexCount;
        mCpuCommandStream[i * 4 + 1] = 1;
        mCpuCommandStream[i * 4 + 2] = mPendingCommands[i].firstIndex;
        mCpuCommandStream[i * 4 + 3] = 0;
    }
    GPU_storagebuf_update(mDrawBuffer, mCpuCommandStream.data());
}

void NaniteRenderer::draw(int viewportWidth, int viewportHeight)
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
    mVisibilityPass.set_data(mLastData == nullptr ? NaniteData() : *mLastData);
    const NaniteViewInfo view = getViewInfo(mLastViewProj, 0.0f, mLastProjScaleY);
    mVisibilityPass.render(mDrawBuffer,
                           mDrawCount,
                           mLastViewProj,
                           view.camPos,
                           viewportWidth,
                           viewportHeight,
                           int(mShadingMode));
}

void NaniteRenderer::getLevelCounts(const NaniteData &data,
                                    const float4x4 &lodViewProj,
                                    float projScaleY,
                                    float viewportHeight,
                                    std::map<uint32_t, uint32_t> &out_counts) const
{
    out_counts.clear();
    std::vector<NaniteDrawCommand> cmds;
    NaniteLodParams params;
    params.pixelError = mPixelError;
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

}  // namespace nanite
}  // namespace blender
