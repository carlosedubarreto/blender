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
    // Reserve a large static capacity for indirect draw commands.
    // Each command is a DrawArraysIndirectCommand (4 x uint32).
    const size_t byteSize = sizeof(uint32_t) * 4 * kDrawCommandCapacity;
    mDrawBuffer = GPU_storagebuf_create(byteSize);
    if (mDrawBuffer == nullptr) {
        return false;
    }
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
    if (mCounterBuf != nullptr) {
        GPU_storagebuf_free(mCounterBuf);
        mCounterBuf = nullptr;
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

void TesseraRenderer::updateWithGpuTraversal(const TesseraData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight)
{
    const size_t clusterCount = data.clusters.size();

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

    /* Upload cluster data only when the mesh changed. */
    size_t indexCount = 0;
    float maxParentError = -1.0f;
    for (const TesseraCluster &c : data.clusters) {
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
                              "tessera_cluster_spheres") ||
            !ensureStorageBuf(&mParentSphereBuf, clusterCount * 4 * sizeof(float),
                              "tessera_cluster_parent_spheres") ||
            !ensureStorageBuf(&mErrorsBuf, clusterCount * 2 * sizeof(float),
                              "tessera_cluster_errors") ||
            !ensureStorageBuf(&mMetaBuf, clusterCount * 4 * sizeof(uint32_t),
                              "tessera_cluster_meta") ||
            !ensureStorageBuf(&mMaterialsBuf, clusterCount * sizeof(uint32_t),
                              "tessera_cluster_materials") ||
            !ensureStorageBuf(&mCounterBuf, kMaxMaterials * sizeof(uint32_t), "tessera_draw_counter"))
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

    /* Reset the shared counter, bind resources and dispatch one thread per
     * cluster. */
    GPU_storagebuf_clear_to_zero(mCounterBuf);

    const TesseraViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);

    GPU_shader_bind(mTraverseShader);
    GPU_storagebuf_bind(mSphereBuf, 0);
    GPU_storagebuf_bind(mParentSphereBuf, 1);
    GPU_storagebuf_bind(mErrorsBuf, 2);
    GPU_storagebuf_bind(mMetaBuf, 3);
    GPU_storagebuf_bind(mDrawBuffer, 4);
    GPU_storagebuf_bind(mCounterBuf, 5);
    GPU_storagebuf_bind(mMaterialsBuf, 6);

    GPU_shader_uniform_3f(mTraverseShader, "camera_pos", view.camPos.x,
                          view.camPos.y, view.camPos.z);
    GPU_shader_uniform_1f(mTraverseShader, "viewport_height", viewportHeight);
    GPU_shader_uniform_1f(mTraverseShader, "proj_scale_y", view.projScaleY);
    GPU_shader_uniform_1f(mTraverseShader, "pixel_error", mPixelError);
    GPU_shader_uniform_1i(mTraverseShader, "cluster_count", int(clusterCount));
    GPU_shader_uniform_1i(mTraverseShader, "max_clusters", 0);
    GPU_shader_uniform_1i(mTraverseShader, "max_commands",
                          int(kDrawCommandCapacity));
    GPU_shader_uniform_1i(mTraverseShader, "max_materials", int(max_materials));
    GPU_shader_uniform_1i(mTraverseShader, "max_commands_per_material", int(mMaxCommandsPerMaterial));
    GPU_shader_uniform_4fv_array(mTraverseShader, "frustum_planes", 6,
                                 view.planes);

    const uint32_t group_size = 64;
    const uint32_t groups = uint32_t((clusterCount + group_size - 1) / group_size);
    GPU_compute_dispatch(mTraverseShader, groups, 1, 1);

    /* Read back the emitted command counters per material. */
    std::vector<uint32_t> emitted(max_materials, 0);
    GPU_storagebuf_read(mCounterBuf, emitted.data());
    mMaterialDrawCounts = std::move(emitted);

    size_t total = 0;
    for (uint32_t c : mMaterialDrawCounts) {
        total += c;
    }
    mDrawCount = total;
    mGpuTraversalActive = true;
}

void TesseraRenderer::updateWithCpuTraversal(const TesseraData &data,
                                            const float4x4 &viewProj,
                                            float projScaleY,
                                            float viewportHeight)
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

    const TesseraViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);
    std::vector<TesseraCluster> selected;

    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const TesseraCluster &cluster = data.clusters[ci];
        if (cluster.indexCount == 0) {
            continue;
        }
        const float3 center(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        const float radius = cluster.sphere[3];
        if (!sphere_in_frustum(view.planes, center, radius)) {
            continue;
        }

        const float err_c = (cluster.error <= 0.0f)
            ? 0.0f
            : projectedErrorPx(center, radius, cluster.error,
                               view.camPos, viewportHeight, view.projScaleY,
                               view.orthographic);

        const float3 pcenter(cluster.parentSphere[0], cluster.parentSphere[1], cluster.parentSphere[2]);
        const float pradius = cluster.parentSphere[3];
        const float err_p = (cluster.parentError >= 1e38f)
            ? std::numeric_limits<float>::max()
            : projectedErrorPx(pcenter, pradius, cluster.parentError,
                               view.camPos, viewportHeight, view.projScaleY,
                               view.orthographic);

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

    if (total == 0) {
        return;
    }

    /* Convert to DrawArraysIndirectCommand stream partitioned per material slot. */
    mCpuCommandStream.assign(kDrawCommandCapacity * 4, 0);
    std::vector<uint32_t> slot_offset(max_materials, 0);
    for (const TesseraCluster &cluster : selected) {
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
                           mMaxCommandsPerMaterial);
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
    mVisibilityPass.render_depth_prepass(mDrawBuffer,
                                         mDrawCount,
                                         mLastViewProj,
                                         viewportWidth,
                                         viewportHeight,
                                         mReverseZ,
                                         &mMaterialDrawCounts,
                                         mMaxCommandsPerMaterial);
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
