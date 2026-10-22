/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  GPU-friendly flattening of the cluster DAG (pure data, no GPU API).
 */

#include "nanite_gpu_data.hh"

#include <vector>

namespace blender {
namespace nanite {

void flattenClustersForGpu(const NaniteData &data, NaniteGpuClusterData &out)
{
    const size_t n = data.clusters.size();
    out.count = n;
    out.spheres.clear();
    out.parent_errors.clear();
    out.meta.clear();
    if (n == 0) {
        return;
    }
    out.spheres.reserve(n * 4);
    out.parent_errors.reserve(n);
    out.meta.reserve(n * 4);

    /* Children counts from the parent links (used for the leaf test). */
    std::vector<uint32_t> children_count(n, 0);
    for (const auto &c : data.clusters) {
        if (c.parentGroup >= 0 && uint32_t(c.parentGroup) < n) {
            children_count[uint32_t(c.parentGroup)]++;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const NaniteCluster &c = data.clusters[i];
        out.spheres.push_back(c.sphere[0]);
        out.spheres.push_back(c.sphere[1]);
        out.spheres.push_back(c.sphere[2]);
        out.spheres.push_back(c.sphere[3]);
        out.parent_errors.push_back(c.parentError);
        out.meta.push_back(c.firstIndex);
        out.meta.push_back(c.indexCount);
        out.meta.push_back(children_count[i]);
        /* Encoded parent index (0xFFFFFFFF for roots) — needed by the GPU
         * traversal for the parent-of-parent test. */
        out.meta.push_back(c.parentGroup >= 0 ? uint32_t(c.parentGroup) :
                                                0xFFFFFFFFu);
    }
}

void flattenVertexDataForGpu(const NaniteData &data, NaniteGpuVertexData &out)
{
    out.vertex_count = data.positions.size();
    out.index_count = data.indices.size();
    out.positions.clear();
    out.normals.clear();
    out.uvs.clear();
    out.indices.clear();
    if (out.vertex_count == 0 || out.index_count == 0) {
        out.vertex_count = 0;
        out.index_count = 0;
        return;
    }

    out.positions.reserve(out.vertex_count * 3);
    out.normals.reserve(out.vertex_count * 3);
    out.uvs.reserve(out.vertex_count * 2);
    out.indices.reserve(out.index_count);

    for (const float3 &p : data.positions) {
        out.positions.push_back(p.x);
        out.positions.push_back(p.y);
        out.positions.push_back(p.z);
    }
    for (const float3 &n : data.normals) {
        out.normals.push_back(n.x);
        out.normals.push_back(n.y);
        out.normals.push_back(n.z);
    }
    const bool has_uvs = !data.uvs.empty();
    for (size_t i = 0; i < out.vertex_count; ++i) {
        if (has_uvs && i < data.uvs.size()) {
            out.uvs.push_back(data.uvs[i].x);
            out.uvs.push_back(data.uvs[i].y);
        }
        else {
            out.uvs.push_back(0.0f);
            out.uvs.push_back(0.0f);
        }
    }
    out.indices = data.indices;
}

}  // namespace nanite
}  // namespace blender
