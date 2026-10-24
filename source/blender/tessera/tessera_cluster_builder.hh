/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Cluster DAG builder.
 *
 *  Implements the Tessera mesh-processing front end: triangle clustering
 *  (region growing), hierarchical merge + quadric simplification, and the
 *  resulting cluster DAG. The finest level ("level 0") clusters partition the
 *  original mesh; each coarser level merges adjacent clusters and simplifies
 *  them with boundary locking so every possible DAG cut renders a watertight
 *  surface.
 */

#pragma once

#include "tessera.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace blender {
namespace tessera {
namespace detail {

struct BuilderParams {
    /** Maximum triangles per cluster (coarser levels may exceed slightly). */
    uint32_t maxClusterTris = 128;
    /** Fraction of triangles kept per simplification step. */
    float simplifyRatio = 0.5f;
    /** Number of clusters merged into one parent group. */
    uint32_t mergeGroupSize = 4;
    /** Safety cap on DAG depth. */
    uint32_t maxLevels = 64;
};

/** Vertex with all attributes needed for the per-cluster private copy. */
struct BuildVertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct BuildTriangle {
    uint32_t v[3];
};

/** One node of the cluster DAG. */
struct BuildCluster {
    /** Triangles of this cluster (indices into #BuildResult::tris). */
    std::vector<uint32_t> tris;
    /** Child clusters (finer level), empty for leaves. */
    std::vector<uint32_t> children;
    float3 sphereCenter;
    float sphereRadius = 0.0f;
    /** Error of the simplification that created this cluster (0 at level 0). */
    float error = 0.0f;
    /**
     * Error of the coarser parent (FLT_MAX for roots). The LOD traversal
     * emits a cluster when this projects below the pixel threshold.
     */
    float parentError = std::numeric_limits<float>::max();
    /** 0 = finest level. */
    uint32_t level = 0;
};

struct BuildResult {
    /** All vertices: the input mesh vertices followed by the private vertex
     * copies appended for coarser-level clusters. */
    std::vector<BuildVertex> verts;
    /** All triangles of all clusters (level 0 first, then parents). */
    std::vector<BuildTriangle> tris;
    /** Cluster nodes, ordered by level ascending (children before parents). */
    std::vector<BuildCluster> clusters;
    uint32_t levelCount = 0;
};

/**
 * Build the cluster DAG from a triangle soup.
 *
 * \param verts      input vertices (referenced by \a tri_verts)
 * \param tri_verts  flat triangle vertex index list into \a verts
 * \param params     builder configuration
 */
BuildResult buildClusterDAG(const std::vector<BuildVertex> &verts,
                            const std::vector<uint32_t> &tri_verts,
                            const BuilderParams &params);

}  // namespace detail
}  // namespace tessera
}  // namespace blender