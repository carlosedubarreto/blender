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

#include "GPU_batch.hh"

namespace blender {
namespace nanite {

bool drawNaniteData(const NaniteData &data,
                    const float4x4 &viewProj,
                    float viewportHeight,
                    gpu::Batch *batch,
                    NaniteRenderer *renderer)
{
    if (batch == nullptr || renderer == nullptr) {
        return false;
    }

    renderer->update(data, viewProj, viewportHeight);

    if (renderer->drawCount() == 0) {
        return true;  // nothing to draw this frame
    }


// ----------------------------------------------------------------------------
// NaniteDrawPass
// ----------------------------------------------------------------------------

NaniteDrawPass::NaniteDrawPass(const NaniteData &data,
                               const float4x4 &viewProj,
                               float viewportHeight)
    : data_(data)
    , view_proj_(viewProj)
    , viewport_height_(viewportHeight)
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
    data_ = data;
    view_proj_ = viewProj;
    viewport_height_ = viewportHeight;
    renderer_.update(data_, view_proj_, viewport_height_);
}

void NaniteDrawPass::draw(gpu::Batch *batch) const
{
    if (batch == nullptr) {
        return;
    }
    renderer_.draw(batch);
}

size_t NaniteDrawPass::drawCount() const
{
    return renderer_.drawCount();
}

