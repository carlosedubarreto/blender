/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  GPU-friendly flattening of the cluster DAG. Pure data, no GPU API
 *  dependencies (consumed by the standalone validation harness too).
 */

#pragma once

#include "nanite.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace blender {
namespace nanite {

/** GPU-friendly per-cluster arrays consumed by the traversal compute shader
 * (std430 layout, base scalar types only, one entry per cluster). */
struct NaniteGpuClusterData {
    /** Spheres: 4 floats per cluster (x, y, z = center, w = radius). */
    std::vector<float> spheres;
    /** Parent spheres: 4 floats per cluster (x, y, z = center, w = radius). */
    std::vector<float> parent_spheres;
    /** Cluster & parent error: 2 floats per cluster (x = error, y = parentError). */
    std::vector<float> errors;
    /** Meta: 4 uints per cluster (firstIndex, indexCount, childrenCount,
     * encodedParentIndex; 0xFFFFFFFF for roots). */
    std::vector<uint32_t> meta;
    size_t count = 0;
};

/**
 * Flatten a #NaniteData into the GPU-friendly per-cluster arrays consumed by
 * the traversal compute shader. \param children_count is derived from the
 * DAG parent links (0 for leaves).
 */
void flattenClustersForGpu(const NaniteData &data, NaniteGpuClusterData &out);

/** Flat vertex/index arrays consumed by the visibility-buffer shaders
 * (std430 layout: plain float / uint arrays, tightly packed). */
struct NaniteGpuVertexData {
    /** 3 floats per vertex (x, y, z). */
    std::vector<float> positions;
    /** 3 floats per vertex. */
    std::vector<float> normals;
    /** 2 floats per vertex (u, v); zeros when the source has no UVs. */
    std::vector<float> uvs;
    /** Index buffer (copy of #NaniteData::indices). */
    std::vector<uint32_t> indices;
    size_t vertex_count = 0;
    size_t index_count = 0;
};

/**
 * Flatten the vertex/index arrays of a #NaniteData for the visibility-buffer
 * passes (vertex pulling + material-pass attribute fetch).
 */
void flattenVertexDataForGpu(const NaniteData &data, NaniteGpuVertexData &out);

}  // namespace nanite
}  // namespace blender
