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
    /** Parent simplification error, 1 float per cluster. */
    std::vector<float> parent_errors;
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

}  // namespace nanite
}  // namespace blender
