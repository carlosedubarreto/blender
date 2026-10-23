/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Draw-pipeline integration implementation.
 */

#include "nanite_draw.hh"

#include "nanite_lod.hh"

#include <cstring>

namespace blender {
namespace nanite {

bool drawNaniteData(const NaniteData &data,
                    const float4x4 &viewProj,
                    float projScaleY,
                    float viewportHeight,
                    int viewportWidth,
                    int viewportHeightPixels,
                    NaniteRenderer *renderer)
{
    if (renderer == nullptr) {
        return false;
    }

    renderer->update(data, viewProj, projScaleY, viewportHeight);

    if (renderer->drawCount() == 0) {
        return true;  // nothing to draw this frame
    }

    renderer->draw(viewportWidth, viewportHeightPixels);
    return true;
}

// ----------------------------------------------------------------------------
// NaniteDrawPass
// ----------------------------------------------------------------------------

NaniteDrawPass::NaniteDrawPass(const NaniteData &data,
                               const float4x4 &viewProj,
                               float projScaleY,
                               float viewportHeight)
    : data_(data), view_proj_(viewProj), viewport_height_(viewportHeight),
      proj_scale_y_(projScaleY)
{
}

NaniteDrawPass::~NaniteDrawPass()
{
    renderer_.terminate();
}

bool NaniteDrawPass::init()
{
    if (!renderer_.init()) {
        return false;
    }
    renderer_.update(data_, view_proj_, proj_scale_y_, viewport_height_);
    return true;
}

void NaniteDrawPass::update(const NaniteData &data,
                            const float4x4 &viewProj,
                            float projScaleY,
                            float viewportHeight,
                            const float4x4 *lodViewProj,
                            const float *lodProjScaleY)
{
    view_proj_ = viewProj;
    proj_scale_y_ = projScaleY;
    viewport_height_ = viewportHeight;

    /* Always forward to renderer::update() — this ensures matrices are
     * fresh each frame and triggers traversal when needed. */
    renderer_.update(data, viewProj, projScaleY, viewportHeight, lodViewProj, lodProjScaleY);

    if (mStreamer != nullptr && mStreamer->isOpen()) {
        const float4x4 &lod_vp = (lodViewProj != nullptr) ? *lodViewProj : view_proj_;
        const float lod_scale = (lodProjScaleY != nullptr) ? *lodProjScaleY : proj_scale_y_;

        /* Streaming mode: data_ holds the assembled resident subset (with
         * stubs for missing pages). The full data was seeded at open time;
         * adopting the caller's full data here would erase the stubs and
         * silence the request collection. */
        if (data_.clusters.size() != mStreamer->clusterCount()) {
            data_ = data; /* first frame: adopt, then immediately stubbed */
        }

        /* Error-driven requests: what would be drawn but is not resident. */
        NaniteLodParams lod_params;
        lod_params.pixelError = 1.0f;
        std::vector<uint32_t> requests;
        collectStreamingRequests(
            data_, lod_vp, lod_scale, viewport_height_, lod_params, requests);
        mStreamer->requestClusters(requests);

        NaniteData assembled;
        mStreamer->update(++mFrame, assembled);
        if (assembled.clusters.size() == mStreamer->clusterCount()) {
            data_ = assembled;
        }
    }
    else {
        data_ = data;
    }
}

bool NaniteDrawPass::openStreaming(const std::string &paged_path,
                                   const NaniteData &full)
{
    if (mStreamer == nullptr) {
        mStreamer = new NanitePageManager();
    }
    if (!mStreamer->open(paged_path)) {
        return false;
    }
    mStreamer->seedRecords(full);
    return true;
}

void NaniteDrawPass::closeStreaming()
{
    if (mStreamer != nullptr) {
        mStreamer->close();
        delete mStreamer;
        mStreamer = nullptr;
    }
}

bool NaniteDrawPass::streamingSettled() const
{
    return mStreamer == nullptr || mStreamer->fullyResident();
}

void NaniteDrawPass::setShadingMode(NaniteShadingMode mode)
{
    renderer_.setShadingMode(mode);
}

void NaniteDrawPass::draw(int viewportWidth, int viewportHeight) const
{
    renderer_.draw(viewportWidth, viewportHeight);
}

size_t NaniteDrawPass::drawCount() const
{
    return renderer_.drawCount();
}

}  // namespace nanite
}  // namespace blender
