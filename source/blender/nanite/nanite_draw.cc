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
                    float viewportHeight,
                    int viewportWidth,
                    int viewportHeightPixels,
                    NaniteRenderer *renderer)
{
    if (renderer == nullptr) {
        return false;
    }

    renderer->update(data, viewProj, viewportHeight);

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
                               float viewportHeight)
    : data_(data), view_proj_(viewProj), viewport_height_(viewportHeight)
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
    renderer_.update(data_, view_proj_, viewport_height_);
    return true;
}

void NaniteDrawPass::update(const NaniteData &data,
                            const float4x4 &viewProj,
                            float viewportHeight)
{
    view_proj_ = viewProj;
    viewport_height_ = viewportHeight;

    if (mStreamer != nullptr && mStreamer->isOpen()) {
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
            data_, view_proj_, viewport_height_, lod_params, requests);
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

    renderer_.update(data_, view_proj_, viewport_height_);
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
