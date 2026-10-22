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

#include <cstddef>

namespace blender {
namespace nanite {

// Convenience wrapper: update the renderer (LOD selection) and render the
// selected clusters with the visibility-buffer pipeline into the currently
// bound framebuffer.
bool drawNaniteData(const NaniteData &data,
                    const float4x4 &viewProj,
                    float viewportHeight,
                    int viewportWidth,
                    int viewportHeightPixels,
                    NaniteRenderer *renderer);

/**
 * Minimal draw pass for Nanite clusters.
 *
 * The pass owns a NaniteRenderer and renders the provided NaniteData with the
 * two-pass visibility-buffer pipeline (visibility rasterization + full-screen
 * material shading). Add this pass to your engine's pass list and call
 * init()/update()/draw() per frame. The draw step renders into the currently
 * bound framebuffer.
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

    void setShadingMode(NaniteShadingMode mode);

    void draw(int viewportWidth, int viewportHeight) const;

    size_t drawCount() const;

private:
    NaniteData data_;
    float4x4 view_proj_;
    float viewport_height_;

    mutable NaniteRenderer renderer_;
};

}  // namespace nanite
}  // namespace blender
