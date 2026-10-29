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

    void set_material_textures(const std::vector<TesseraMaterialTextures> &materials)
    {
        renderer_.set_material_textures(materials);
    }

    void setTextureScale(float scale)
    {
        renderer_.setTextureScale(scale);
    }

    float getTextureScale() const
    {
        return renderer_.getTextureScale();
    }

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

    void setCullingMargin(float margin)
    {
        renderer_.setCullingMargin(margin);
    }

    float getCullingMargin() const
    {
        return renderer_.getCullingMargin();
    }

    void setReverseZ(bool enabled)
    {
        renderer_.setReverseZ(enabled);
    }

    bool getReverseZ() const
    {
        return renderer_.getReverseZ();
    }

    void setTelemetryEnabled(bool enabled)
    {
        renderer_.setTelemetryEnabled(enabled);
    }

    bool isTelemetryEnabled() const
    {
        return renderer_.isTelemetryEnabled();
    }

    const TesseraTelemetry &getTelemetry() const
    {
        return renderer_.getTelemetry();
    }

    void setNavigating(bool navigating)
    {
        renderer_.setNavigating(navigating);
    }

    bool isNavigating() const
    {
        return renderer_.isNavigating();
    }

    void getLevelCounts(const TesseraData &data,
                        const float4x4 &lodViewProj,
                        float projScaleY,
                        float viewportHeight,
                        std::map<uint32_t, uint32_t> &out_counts) const
    {
        renderer_.getLevelCounts(data, lodViewProj, projScaleY, viewportHeight, out_counts);
    }

    void draw(int viewportWidth, int viewportHeight,
              const float4x4 *view_proj_override = nullptr,
              const float4x4 *object_model_override = nullptr) const;
    void draw_prepass(int viewportWidth, int viewportHeight,
                      const float4x4 *view_proj_override = nullptr) const;
    void draw_gbuffer(int viewportWidth, int viewportHeight,
                      const float4x4 &object_model = float4x4::identity(),
                      const float4x4 *view_proj_override = nullptr) const;
    void draw_forward(const float4x4 &model_mat,
                      int viewportWidth,
                      int viewportHeight,
                      int lighting_type = 1,
                      const float4 &base_color = float4(0.8f, 0.8f, 0.8f, 1.0f),
                      const float4x4 *view_proj_override = nullptr) const;

    /** Two-sided / foliage-card mode (see TesseraVisibilityPass). */
    void set_two_sided(bool two_sided)
    {
        renderer_.set_two_sided(two_sided);
    }

    /** Alpha cutout mode (see TesseraVisibilityPass). */
    void set_alpha_mask(bool enabled)
    {
        renderer_.set_alpha_mask(enabled);
    }

    size_t drawCount(uint32_t material_index = 0) const;
    size_t totalDrawCount() const;
    size_t materialCommandOffset(uint32_t material_index) const;
    const std::vector<uint32_t> &materialDrawCounts() const;
    uint32_t maxCommandsPerMaterial() const;

    /** Enable streaming from a paged .nc2 file. */
    bool openStreaming(const std::string &paged_path, const TesseraData &full);
    void closeStreaming();
    bool streaming() const { return mStreamer != nullptr; }
    /** True when no pages are missing anymore. */
    bool streamingSettled() const;

    /** Memory budget & streaming telemetry. */
    TesseraMemoryBudget getMemoryBudget() const;
    void setBudget(size_t bytes);
    void setInstanceCounts(uint32_t total, uint32_t active);
    void prefetchClusters(const std::vector<uint32_t> &cluster_ids);

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
