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

}  // namespace nanite
}  // namespace blender
