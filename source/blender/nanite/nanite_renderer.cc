/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  NaniteRenderer implementation.
 */

#include "nanite_renderer.hh"

#include "nanite_lod.hh"

#include "GPU_batch.hh"
#include "GPU_storage_buffer.hh"

namespace blender {
namespace nanite {

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

void NaniteRenderer::terminate()
{
    if (mDrawBuffer != nullptr) {
        GPU_storagebuf_free(mDrawBuffer);
        mDrawBuffer = nullptr;
    }
    mPendingCommands.clear();
    mDrawCount = 0;
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

    // Run LOD selection (CPU). The viewProj is accepted for future GPU-based
    // selection but is not required by the current CPU implementation.
    (void)viewProj;
    size_t count = selectClusters(
        data, viewProj, viewportHeight, NaniteLodParams(), mPendingCommands);
    mDrawCount = count;

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
