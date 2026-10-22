/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  GPU renderer: LOD selection on the CPU, indirect multi-draw on the GPU.
 *
 *  The renderer owns a GPU storage buffer that holds the indirect draw
 *  commands produced by selectClusters. Draw calls are issued through
 *  Blender's GPU batch abstraction (GPU_batch_multi_draw_indirect).
 */

#pragma once

#include "nanite.h"

#include "GPU_batch.hh"
#include "GPU_storage_buffer.hh"

#include <cstddef>
#include <vector>

namespace blender {
namespace nanite {

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

    /**
     * Call once per frame to issue the draw call.
     * \param batch The gpu::Batch describing the mesh vertex/index buffers.
     *              The batch must be configured before calling this function.
     */
    void draw(gpu::Batch *batch);

    size_t drawCount() const { return mDrawCount; }

private:
    // Staging host-side command list produced by selectClusters.
    std::vector<NaniteDrawCommand> mPendingCommands;

    // GPU storage buffer holding the indirect draw commands (std430 layout).
    gpu::StorageBuf *mDrawBuffer = nullptr;

    // Number of draw calls written this frame.
    size_t mDrawCount = 0;

    // Constant host-side capacity. Granularity is kept to avoid reallocation.
    static constexpr size_t kDrawCommandCapacity = 2000000;  // 40 MB
};

}  // namespace nanite
}  // namespace blender
