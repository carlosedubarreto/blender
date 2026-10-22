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
    // Reserve a large static capacity for indirect draw commands (std430).
    const size_t byteSize = sizeof(NaniteDrawCommand) * kDrawCommandCapacity;
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
    if (mParentErrBuf != nullptr) {
        GPU_storagebuf_free(mParentErrBuf);
        mParentErrBuf = nullptr;
    }
    if (mMetaBuf != nullptr) {
        GPU_storagebuf_free(mMetaBuf);
        mMetaBuf = nullptr;
    }
    if (mCounterBuf != nullptr) {
        GPU_storagebuf_free(mCounterBuf);
        mCounterBuf = nullptr;
    }
    mGpuClusters = NaniteGpuClusterData();
    mPendingCommands.clear();
    mDrawCount = 0;
    mGpuTraversalActive = false;
    mGpuPathFailed = false;
    mUploadedClusterCount = size_t(-1);
    mUploadedIndexCount = size_t(-1);
    mUploadedMaxParentError = -1.0f;
}

void NaniteRenderer::update(const NaniteData &data,
                            const float4x4 &viewProj,
                            float viewportHeight)
{
    mPendingCommands.clear();
    mDrawCount = 0;

    if (data.clusters.empty() || mDrawBuffer == nullptr) {
        return;
    }

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
        updateWithGpuTraversal(data, viewProj, viewportHeight);
    }
    else {
        mGpuPathFailed = true;
        updateWithCpuTraversal(data, viewProj, viewportHeight);
    }
}

void NaniteRenderer::updateWithGpuTraversal(const NaniteData &data,
                                            const float4x4 &viewProj,
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
            !ensureStorageBuf(&mParentErrBuf, clusterCount * sizeof(float),
                              "nanite_cluster_parent_err") ||
            !ensureStorageBuf(&mMetaBuf, clusterCount * 4 * sizeof(uint32_t),
                              "nanite_cluster_meta") ||
            !ensureStorageBuf(&mCounterBuf, sizeof(uint32_t), "nanite_draw_counter"))
        {
            mGpuPathFailed = true;
            return updateWithCpuTraversal(data, viewProj, viewportHeight);
        }
        GPU_storagebuf_update(mSphereBuf, mGpuClusters.spheres.data());
        GPU_storagebuf_update(mParentErrBuf, mGpuClusters.parent_errors.data());
        GPU_storagebuf_update(mMetaBuf, mGpuClusters.meta.data());
        mUploadedClusterCount = clusterCount;
        mUploadedIndexCount = indexCount;
        mUploadedMaxParentError = maxParentError;
    }

    /* Reset the shared counter, bind resources and dispatch one thread per
     * cluster. */
    GPU_storagebuf_clear_to_zero(mCounterBuf);

    const NaniteViewInfo view = getViewInfo(viewProj, viewportHeight);

    GPU_shader_bind(mTraverseShader);
    GPU_storagebuf_bind(mSphereBuf, 0);
    GPU_storagebuf_bind(mParentErrBuf, 1);
    GPU_storagebuf_bind(mMetaBuf, 2);
    GPU_storagebuf_bind(mDrawBuffer, 3);
    GPU_storagebuf_bind(mCounterBuf, 4);

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
                                            float viewportHeight)
{
    NaniteLodParams params;
    params.pixelError = mPixelError;
    const size_t count = selectClustersFlat(
        data, viewProj, viewportHeight, params, mPendingCommands);
    mDrawCount = count;
    mGpuTraversalActive = false;

    if (count == 0) {
        return;
    }

    // Upload the draw commands to the GPU storage buffer.
    // The buffer was allocated with enough capacity (kDrawCommandCapacity),
    // so a direct memcpy-like upload is safe.
    GPU_storagebuf_update(mDrawBuffer, mPendingCommands.data());
}

void NaniteRenderer::draw(gpu::Batch *batch)
{
    if (batch == nullptr || mDrawBuffer == nullptr) {
        return;
    }

    // Ensure the host-visible storage buffer is synchronized as an indirect
    // buffer before issuing the draw.
    GPU_storagebuf_sync_as_indirect_buffer(mDrawBuffer);

    const int count = static_cast<int>(mDrawCount);
    if (count <= 0) {
        return;
    }

    // Stride is sizeof(NaniteDrawCommand) (20 bytes), matching the
    // VkDrawIndexedIndirectCommand layout.
    GPU_batch_multi_draw_indirect(batch, mDrawBuffer, count, 0,
                                   static_cast<intptr_t>(sizeof(NaniteDrawCommand)));
}

}  // namespace nanite
}  // namespace blender
