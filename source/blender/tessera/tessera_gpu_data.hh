/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  GPU-friendly flattening of the cluster DAG. Pure data, no GPU API
 *  dependencies (consumed by the standalone validation harness too).
 */

#pragma once

#include "tessera.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace blender {
namespace tessera {

/** GPU-friendly per-cluster arrays consumed by the traversal compute shader
 * (std430 layout, base scalar types only, one entry per cluster). */
struct TesseraGpuClusterData {
    /** Spheres: 4 floats per cluster (x, y, z = center, w = radius). */
    std::vector<float> spheres;
    /** Parent spheres: 4 floats per cluster (x, y, z = center, w = radius). */
    std::vector<float> parent_spheres;
    /** Cluster & parent error: 2 floats per cluster (x = error, y = parentError). */
    std::vector<float> errors;
    /** Meta: 4 uints per cluster (firstIndex, indexCount, childrenCount,
     * encodedParentIndex; 0xFFFFFFFF for roots). */
    std::vector<uint32_t> meta;
    /** Material slot index per cluster. */
    std::vector<uint32_t> materials;
    size_t count = 0;
};

/**
 * Flatten a #TesseraData into the GPU-friendly per-cluster arrays consumed by
 * the traversal compute shader. \param children_count is derived from the
 * DAG parent links (0 for leaves).
 */
void flattenClustersForGpu(const TesseraData &data, TesseraGpuClusterData &out);

/** Flat vertex/index arrays consumed by the visibility-buffer shaders
 * (std430 layout: plain float / uint arrays, tightly packed). */
struct TesseraGpuVertexData {
    /** 3 floats per vertex (x, y, z). */
    std::vector<float> positions;
    /** 3 floats per vertex. */
    std::vector<float> normals;
    /** 2 floats per vertex (u, v); zeros when the source has no UVs. */
    std::vector<float> uvs;
    /** 4 floats per vertex (tangent xyz, sign w); default (1,0,0,1). */
    std::vector<float> tangents;
    /** Index buffer (copy of #TesseraData::indices). */
    std::vector<uint32_t> indices;
    size_t vertex_count = 0;
    size_t index_count = 0;
};

/**
 * Flatten the vertex/index arrays of a #TesseraData for the visibility-buffer
 * passes (vertex pulling + material-pass attribute fetch).
 */
void flattenVertexDataForGpu(const TesseraData &data, TesseraGpuVertexData &out);

}  // namespace tessera
}  // namespace blender
