/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  CPU LOD selection for Tessera clusters.
 *
 *  Implements a projected screen-space error metric: a cluster is kept when
 *  its world-space simplification error scaled by the viewport height and the
 *  cluster sphere radius falls below the configured pixel threshold.
 */

#pragma once

#include "tessera.h"

#include "BLI_math_matrix_types.hh"
#include "BLI_math_vector_types.hh"

#include <cstddef>
#include <vector>

namespace blender {
namespace tessera {

/** Per-view values shared by the CPU and GPU traversal paths. */
struct TesseraViewInfo {
    /** World-space camera position (perspective) or origin (orthographic). */
    float3 camPos;
    /** Six frustum planes (a*x + b*y + c*z + d >= 0 inside), normalized. */
    float planes[6][4];
    /** Vertical projection scale: proj[1][1]. */
    float projScaleY = 0.0f;
    /** True for orthographic views (no eye point; unit-distance metric). */
    bool orthographic = false;
};

/**
 * Extract the frustum planes, camera position and projection scale from a
 * view-projection matrix. Shared by the recursive CPU traversal, the flat CPU
 * traversal and the GPU compute traversal.
 *
 * \\note \\a projScaleY must come from the projection matrix alone, not from
 * the combined view-projection matrix: when the object transform is baked into
 * viewProj (as the draw engine does so clusters can stay in object space),
 * viewProj[1][1] is corrupted by object rotation and no longer equals the
 * vertical field-of-view scale.
 */
TesseraViewInfo getViewInfo(const float4x4 &viewProj,
                           float viewportHeight,
                           float projScaleY);

/** Sphere vs. frustum test (Gribb-Hartmann planes, normalized). */
bool sphere_in_frustum(const float planes[6][4], const float3 &center,
                       float radius);

/**
 * Project a world-space error value to screen pixels, accounting for the
 * distance of the camera to the cluster (perspective-correct metric).
 */
float projectedErrorPx(const float3 &center, float radius, float error,
                       const float3 &camPos, float viewportHeight,
                       float projScaleY, bool orthographic);

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
std::size_t selectClusters(const TesseraData &data,
                           const float4x4 &viewProj,
                           float projScaleY,
                           float viewportHeight,
                           const TesseraLodParams &params,
                           std::vector<TesseraDrawCommand> &out);

/**
 * Flat (fully parallelizable) variant of the traversal rule: a cluster is
 * drawn when it has no children, or when the error of its coarser parent
 * projects below the pixel threshold. Produces the same draw set as
 * #selectClusters (when the cluster budget is disabled) but requires no DAG
 * recursion — every cluster can be evaluated independently. This is the CPU
 * mirror of the GPU compute traversal and doubles as the CPU fallback.
 */
std::size_t selectClustersFlat(const TesseraData &data,
                               const float4x4 &viewProj,
                               float projScaleY,
                               float viewportHeight,
                               const TesseraLodParams &params,
                               std::vector<TesseraDrawCommand> &out);

/**
 * Approximate the screen-space error of a cluster in pixels.
 *
 * Simple metric: world error scaled by viewport height and cluster radius.
 * A cluster whose simplification error is small relative to its radius and
 * to the current viewport height produces a small projected error.
 */
float clusterScreenError(const TesseraCluster &cluster, float viewportHeight);

}  // namespace tessera
}  // namespace blender
