/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  LOD selection implementation.
 */

#include "nanite_lod.hh"

#include <cfloat>
#include <limits>
#include <vector>

namespace blender {
namespace nanite {

float clusterScreenError(const NaniteCluster &cluster, float viewportHeight)
{
    const float radius = cluster.sphere[3];
    if (radius <= 0.0f) {
        return std::numeric_limits<float>::max();
    }
    // Scale the world error by the ratio of the sphere radius to the viewport
    // height so the value is a proxy for screen-space pixels.
    return (cluster.error / radius) * (viewportHeight * 0.5f);
}

size_t selectClusters(const NaniteData &data,
                      const float4x4 & /*viewProj*/,
                      float viewportHeight,
                      const NaniteLodParams &params,
                      std::vector<NaniteDrawCommand> &out)
{
    out.clear();

    if (data.clusters.empty()) {
        return 0;
    }

    const uint32_t maxClusters = params.maxClustersPerFrame > 0
                                     ? params.maxClustersPerFrame
                                     : std::numeric_limits<uint32_t>::max();

    for (const NaniteCluster &cluster : data.clusters) {
        const float screenError = clusterScreenError(cluster, viewportHeight);
        if (screenError <= params.pixelError) {
            if (out.size() >= maxClusters) {
                break;
            }
            NaniteDrawCommand cmd = {};
            cmd.indexCount = cluster.indexCount;
            cmd.instanceCount = 1;
            cmd.firstIndex = cluster.firstIndex;
            cmd.vertexOffset = 0;  // vertexOffset is relative to the vertex buffer
            cmd.firstInstance = 0;
            out.push_back(cmd);
        }
    }

    return out.size();
}

}  // namespace nanite
}  // namespace blender
