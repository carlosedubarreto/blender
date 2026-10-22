/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  CPU LOD selection for Nanite clusters.
 *
 *  Implements a projected screen-space error metric: a cluster is kept when
 *  its world-space simplification error scaled by the viewport height and the
 *  cluster sphere radius falls below the configured pixel threshold.
 */

#pragma once

#include "nanite.h"

#include <cstddef>
#include <vector>

namespace blender {
namespace nanite {

/**
 * Select clusters to draw for a given camera view and viewport height.
 *
 * The output is a flat list of draw commands (one per selected cluster). The
 * caller is responsible for uploading this list to a GPU storage buffer and
 * issuing a multi-draw-indirect call.
 *
 * \param data        cluster data (from buildFromMesh / serialize)
 * \param viewProj    4x4 view-projection matrix (world -> clip)
 * \param viewportHeight height of the viewport in pixels (non-zero)
 * \param params      LOD parameters (pixel error threshold + optional cap)
 * \param out         output draw command list
 * \returns number of draw commands written
 */
size_t selectClusters(const NaniteData &data,
                      const float4x4 &viewProj,
                      float viewportHeight,
                      const NaniteLodParams &params,
                      std::vector<NaniteDrawCommand> &out);

/**
 * Approximate the screen-space error of a cluster in pixels.
 *
 * Simple metric: world error scaled by viewport height and cluster radius.
 * A cluster whose simplification error is small relative to its radius and
 * to the current viewport height produces a small projected error.
 */
float clusterScreenError(const NaniteCluster &cluster, float viewportHeight);

}  // namespace nanite
}  // namespace blender
