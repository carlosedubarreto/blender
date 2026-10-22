/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  GPU renderer: phase 1 — GPU-driven traversal, indirect multi-draw.
 *
 *  The renderer owns the indirect draw command storage buffer, the per-cluster
 *  data buffers and the traversal compute shader. Draw calls are issued
 *  through Blender's GPU batch abstraction (GPU_batch_multi_draw_indirect).
 */

#pragma once

#include "nanite.h"
#include "nanite_gpu_traverse.hh"

#include "GPU_batch.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace blender {
namespace nanite {

/**
 * GPU renderer: phase 1 — fully GPU-driven traversal.
 *
 * Every frame a single compute-shader dispatch performs frustum culling and
 * LOD selection for all clusters in parallel and atomically appends indirect
 * draw commands into a storage buffer. The only CPU work left is one
 * dispatch, one 4-byte counter readback and (only when the mesh changes) one
 * cluster-data upload. Draw calls are issued through Blender's GPU batch
 * abstraction (GPU_batch_multi_draw_indirect).
 *
 * When compute shaders are unavailable the renderer transparently falls back
 * to the validated CPU traversal (selectClustersFlat) and uploads the result.
 */
class NaniteRenderer
{
public:
    NaniteRenderer() = default;
    ~NaniteRenderer();

    NaniteRenderer(const NaniteRenderer&) = delete;
    NaniteRenderer& operator=(const NaniteRenderer&) = delete;

    /** Initialize the renderer (create the indirect draw buffer). */
    bool init();

    /** Release GPU resources. */
    void terminate();

    /** Call once per frame to recompute LOD selection and update the command buffer. */
    void update(const NaniteData &data,
                const float4x4 &viewProj,
                float viewportHeight);

    /** Override the LOD pixel error threshold for the next frames. */
    void setPixelError(float pixelError) { mPixelError = pixelError; }

    /**
     * Call once per frame to issue the draw call.
     * \param batch The gpu::Batch describing the mesh vertex/index buffers.
     *              The batch must be configured before calling this function.
     */
    void draw(gpu::Batch *batch);

    size_t drawCount() const { return mDrawCount; }

    /** True when the GPU compute traversal is in use. */
    bool usesGpuTraversal() const { return mGpuTraversalActive; }

private:
    /** CPU fallback: run the flat traversal and upload the commands. */
    void updateWithCpuTraversal(const NaniteData &data,
                                const float4x4 &viewProj,
                                float viewportHeight);

    /** GPU path: upload cluster data (if changed), dispatch, read back count. */
    void updateWithGpuTraversal(const NaniteData &data,
                                const float4x4 &viewProj,
                                float viewportHeight);

    /** (Re)create or resize one SSBO to at least \a byteSize. */
    bool ensureStorageBuf(gpu::StorageBuf **buf, size_t byteSize, const char *name);

    // Staging host-side command list produced by the CPU fallback path.
    std::vector<NaniteDrawCommand> mPendingCommands;

    // GPU storage buffer holding the indirect draw commands (std430 layout).
    gpu::StorageBuf *mDrawBuffer = nullptr;

    // Compute shader implementing the traversal (created lazily).
    gpu::Shader *mTraverseShader = nullptr;

    // Per-cluster data buffers (uploaded only when the mesh changes).
    gpu::StorageBuf *mSphereBuf = nullptr;
    gpu::StorageBuf *mParentErrBuf = nullptr;
    gpu::StorageBuf *mMetaBuf = nullptr;
    // Shared atomic counter written by the compute shader, read back on CPU.
    gpu::StorageBuf *mCounterBuf = nullptr;

    NaniteGpuClusterData mGpuClusters;

    // Fingerprint of the last uploaded cluster data (dirty tracking).
    size_t mUploadedClusterCount = size_t(-1);
    size_t mUploadedIndexCount = size_t(-1);
    float mUploadedMaxParentError = -1.0f;

    // Number of draw calls written this frame.
    size_t mDrawCount = 0;

    // LOD pixel error threshold (pixels).
    float mPixelError = 1.0f;

    // Set once the GPU path is proven working; set permanently if the
    // compute shader could not be created.
    bool mGpuTraversalActive = false;
    bool mGpuPathFailed = false;

    // Constant host-side capacity. Granularity is kept to avoid reallocation.
    static constexpr size_t kDrawCommandCapacity = 2000000;  // 40 MB
};

}  // namespace nanite
}  // namespace blender
