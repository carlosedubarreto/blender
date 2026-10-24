/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Quadric error metric (QEM) edge-collapse simplification.
 *
 *  Operates on a local triangle set (positions + flat triangle vertex list).
 *  Vertices lying on the open boundary of the set are locked, so the border
 *  of a simplified cluster patch stays exactly where it was. This is what
 *  keeps parent and child clusters watertight-compatible in the cluster DAG:
 *  any cut through the DAG selects only boundary-matched geometry.
 */

#pragma once

#include "tessera.h"

#include <cstdint>
#include <vector>

namespace blender {
namespace tessera {
namespace detail {

struct SimplifyResult {
    /**
     * Vertex positions of the simplified set (compact local array; triangles
     * below reference indices into this array). Positions of locked boundary
     * vertices are unchanged; interior vertices may have moved.
     */
    std::vector<float3> positions;
    /** Simplified triangle list (flat, 3 local vertex indices per triangle). */
    std::vector<uint32_t> tris;
    /**
     * Maximum geometric error (world units) introduced by the collapses
     * performed. Zero when no collapse was possible.
     */
    float error = 0.0f;
};

/**
 * Simplify a triangle set down to \a target_tri_count triangles using
 * iterative edge collapses ranked by quadric error.
 *
 * \param positions       vertex positions referenced by \a tri_verts
 *                        (the array is not modified; simplification works on
 *                        a local copy)
 * \param tri_verts       flat triangle vertex index list
 * \param target_tri_count stop collapsing once at or below this count
 */
SimplifyResult simplifyTriangles(const std::vector<float3> &positions,
                                 const std::vector<uint32_t> &tri_verts,
                                 uint32_t target_tri_count);

}  // namespace detail
}  // namespace tessera
}  // namespace blender