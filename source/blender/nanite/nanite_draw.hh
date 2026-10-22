/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Draw-pipeline integration declarations.
 */

#pragma once

#include "nanite.h"
#include "nanite_renderer.hh"

#include "GPU_batch.hh"

#include <cstddef>

namespace blender {
namespace nanite {

// Convenience wrapper: update the renderer (LOD selection) and issue the
// indirect draw command for a NaniteData buffer.
bool drawNaniteData(const NaniteData &data,
                    const float4x4 &viewProj,
                    float viewportHeight,
                    gpu::Batch *batch,
                    NaniteRenderer *renderer);

/**
 * Minimal draw pass for Nanite clusters.
 *
 * The pass owns a NaniteRenderer and renders the provided NaniteData with a
 * single indirect-multi-draw call, exactly like the Nanite_CUP2 viewer. Add
 * this pass to your engine's pass list and call init()/draw() per frame.
 */
class NaniteDrawPass
{
public:
    NaniteDrawPass(const NaniteData &data,
                   const float4x4 &viewProj,
                   float viewportHeight);
    ~NaniteDrawPass();

    NaniteDrawPass(const NaniteDrawPass&) = delete;
    NaniteDrawPass& operator=(const NaniteDrawPass&) = delete;

    bool init();

    void update(const NaniteData &data,
                const float4x4 &viewProj,
                float viewportHeight);

    void draw(gpu::Batch *batch) const;

    size_t drawCount() const;

private:
    NaniteData data_;
    float4x4 view_proj_;
    float viewport_height_;

    mutable NaniteRenderer renderer_;
};

}  // namespace nanite
}  // namespace blender
