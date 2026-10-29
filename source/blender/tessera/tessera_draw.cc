/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Draw-pipeline integration implementation.
 */

#include "tessera_draw.hh"

#include "tessera_lod.hh"

#include <cstring>

namespace blender {
namespace tessera {

bool drawTesseraData(const TesseraData &data,
                    const float4x4 &viewProj,
                    float projScaleY,
                    float viewportHeight,
                    int viewportWidth,
                    int viewportHeightPixels,
                    TesseraRenderer *renderer)
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
// TesseraDrawPass
// ----------------------------------------------------------------------------

TesseraDrawPass::TesseraDrawPass(const TesseraData &data,
                               const float4x4 &viewProj,
                               float projScaleY,
                               float viewportHeight)
    : data_(data), view_proj_(viewProj), viewport_height_(viewportHeight),
      proj_scale_y_(projScaleY)
{
}

TesseraDrawPass::~TesseraDrawPass()
{
    renderer_.terminate();
}

bool TesseraDrawPass::init()
{
    if (!renderer_.init()) {
        return false;
    }
    renderer_.update(data_, view_proj_, proj_scale_y_, viewport_height_);
    return true;
}

void TesseraDrawPass::update(const TesseraData &data,
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
        TesseraLodParams lod_params;
        lod_params.pixelError = 1.0f;
        std::vector<uint32_t> requests;
        collectStreamingRequests(
            data_, lod_vp, lod_scale, viewport_height_, lod_params, requests);
        mStreamer->requestClusters(requests);

        TesseraData assembled;
        mStreamer->update(++mFrame, assembled);
        if (assembled.clusters.size() == mStreamer->clusterCount()) {
            data_ = assembled;
        }
    }
    else {
        if (&data_ != &data && (data_.clusters.size() != data.clusters.size() ||
                                data_.indices.size() != data.indices.size() ||
                                data_.positions.size() != data.positions.size())) {
            data_ = data;
        }
    }
}

bool TesseraDrawPass::openStreaming(const std::string &paged_path,
                                   const TesseraData &full)
{
    if (mStreamer == nullptr) {
        mStreamer = new TesseraPageManager();
    }
    if (!mStreamer->open(paged_path)) {
        return false;
    }
    mStreamer->seedRecords(full);
    return true;
}

void TesseraDrawPass::closeStreaming()
{
    if (mStreamer != nullptr) {
        mStreamer->close();
        delete mStreamer;
        mStreamer = nullptr;
    }
}

bool TesseraDrawPass::streamingSettled() const
{
    return mStreamer == nullptr || mStreamer->fullyResident();
}

TesseraMemoryBudget TesseraDrawPass::getMemoryBudget() const
{
    if (mStreamer != nullptr) {
        return mStreamer->getMemoryBudget();
    }
    TesseraMemoryBudget budget;
    budget.current_resident_bytes = data_.positions.size() * sizeof(float) +
                                    data_.indices.size() * sizeof(uint32_t) +
                                    data_.clusters.size() * sizeof(TesseraCluster);
    budget.max_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
    return budget;
}

void TesseraDrawPass::setBudget(size_t bytes)
{
    if (mStreamer != nullptr) {
        mStreamer->setBudget(bytes);
    }
}

void TesseraDrawPass::setInstanceCounts(uint32_t total, uint32_t active)
{
    if (mStreamer != nullptr) {
        mStreamer->setInstanceCounts(total, active);
    }
}

void TesseraDrawPass::prefetchClusters(const std::vector<uint32_t> &cluster_ids)
{
    if (mStreamer != nullptr) {
        mStreamer->prefetchClusters(cluster_ids);
    }
}

void TesseraDrawPass::setShadingMode(TesseraShadingMode mode)
{
    renderer_.setShadingMode(mode);
}

void TesseraDrawPass::draw(int viewportWidth, int viewportHeight,
                           const float4x4 *view_proj_override,
                           const float4x4 *object_model_override) const
{
    renderer_.draw(viewportWidth, viewportHeight, view_proj_override, object_model_override);
}

void TesseraDrawPass::draw_prepass(int viewportWidth, int viewportHeight,
                                   const float4x4 *view_proj_override) const
{
    renderer_.draw_prepass(viewportWidth, viewportHeight, view_proj_override);
}

void TesseraDrawPass::draw_gbuffer(int viewportWidth, int viewportHeight, const float4x4 &object_model,
                                   const float4x4 *view_proj_override) const
{
    renderer_.draw_gbuffer(viewportWidth, viewportHeight, object_model, view_proj_override);
}

void TesseraDrawPass::draw_forward(const float4x4 &model_mat,
                                   int viewportWidth,
                                   int viewportHeight,
                                   int lighting_type,
                                   const float4 &base_color,
                                   const float4x4 *view_proj_override) const
{
    renderer_.draw_forward(model_mat, viewportWidth, viewportHeight, lighting_type, base_color, view_proj_override);
}

size_t TesseraDrawPass::drawCount(uint32_t material_index) const
{
    return renderer_.drawCount(material_index);
}

size_t TesseraDrawPass::totalDrawCount() const
{
    return renderer_.totalDrawCount();
}

size_t TesseraDrawPass::materialCommandOffset(uint32_t material_index) const
{
    return renderer_.materialCommandOffset(material_index);
}

const std::vector<uint32_t> &TesseraDrawPass::materialDrawCounts() const
{
    return renderer_.materialDrawCounts();
}

uint32_t TesseraDrawPass::maxCommandsPerMaterial() const
{
    return renderer_.maxCommandsPerMaterial();
}

}  // namespace tessera
}  // namespace blender
