/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Draw-pipeline integration declarations.
 */

#pragma once

#include "tessera.h"
#include "tessera_paging.hh"
#include "tessera_renderer.hh"

#include <cstdint>
#include <map>
#include <string>

#include <cstddef>

namespace blender {
namespace tessera {

// Convenience wrapper: update the renderer (LOD selection) and render the
// selected clusters with the visibility-buffer pipeline into the currently
// bound framebuffer.
bool drawTesseraData(const TesseraData &data,
                    const float4x4 &viewProj,
                    float viewportHeight,
                    int viewportWidth,
                    int viewportHeightPixels,
                    TesseraRenderer *renderer);

/**
 * Minimal draw pass for Tessera clusters.
 *
 * The pass owns a TesseraRenderer and renders the provided TesseraData with the
 * two-pass visibility-buffer pipeline (visibility rasterization + full-screen
 * material shading). Add this pass to your engine's pass list and call
 * init()/update()/draw() per frame. The draw step renders into the currently
 * bound framebuffer.
 *
 * Optionally the pass drives a #TesseraPageManager: when streaming is open,
 * update() collects error-driven requests from the last frame, pumps the
 * async loader and renders the resident subset (missing pages appear as
 * stubs and are simply not drawn until loaded).
 */
class TesseraDrawPass
{
public:
    TesseraDrawPass(const TesseraData &data,
                   const float4x4 &viewProj,
                   float projScaleY,
                   float viewportHeight);
    ~TesseraDrawPass();

    TesseraDrawPass(const TesseraDrawPass&) = delete;
    TesseraDrawPass& operator=(const TesseraDrawPass&) = delete;

    bool init();

    void update(const TesseraData &data,
                const float4x4 &viewProj,
                float projScaleY,
                float viewportHeight,
                const float4x4 *lodViewProj = nullptr,
                const float *lodProjScaleY = nullptr);

    void setShadingMode(TesseraShadingMode mode);

    /**
     * Screen-space error threshold in pixels for LOD selection.
     * Higher values select coarser clusters (faster, less detail).
     */
    void setPixelError(float pixelError)
    {
        renderer_.setPixelError(pixelError);
    }

    float getPixelError() const
    {
        return renderer_.getPixelError();
    }

    void getLevelCounts(const TesseraData &data,
                        const float4x4 &lodViewProj,
                        float projScaleY,
                        float viewportHeight,
                        std::map<uint32_t, uint32_t> &out_counts) const
    {
        renderer_.getLevelCounts(data, lodViewProj, projScaleY, viewportHeight, out_counts);
    }

    void draw(int viewportWidth, int viewportHeight) const;

    size_t drawCount() const;

    /** Enable streaming from a paged .nc2 file. */
    bool openStreaming(const std::string &paged_path, const TesseraData &full);
    void closeStreaming();
    bool streaming() const { return mStreamer != nullptr; }
    /** True when no pages are missing anymore. */
    bool streamingSettled() const;

private:
    TesseraData data_;
    float4x4 view_proj_;
    float viewport_height_;
    float proj_scale_y_;

    mutable TesseraRenderer renderer_;
    TesseraPageManager *mStreamer = nullptr;
    uint64_t mFrame = 0;
};

}  // namespace tessera
}  // namespace blender
