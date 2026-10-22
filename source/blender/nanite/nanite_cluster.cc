/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Cluster builder and serialization implementation.
 */

#include "nanite_cluster.hh"

#include "BKE_mesh.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace blender {
namespace nanite {
namespace detail {

void computeTriangleSphere(const float3 &a, const float3 &b, const float3 &c,
                           float3 &center, float &radius)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float dAB = lengthSquared(ab);
    const float dAC = lengthSquared(ac);
    const float dBC = lengthSquared(bc);
    const float3 &e1 = (dAB >= dAC) ? ((dAB >= dBC) ? ab : bc)
                                     : ((dAC >= dBC) ? ac : bc);
    const float3 &e2 = (dAB >= dAC) ? ((dAB >= dBC) ? ac : bc)
                                     : ((dAC >= dBC) ? ab : bc);
    const float3 mid = (e1 + e2) * 0.5f;
    float len = length(e2 - e1);
    len = len > 0.0f ? len : 1.0f;
    center = mid;
    radius = length(e2 - mid);
    float d = length(a - center);
    if (d > radius) { radius = d; }
    d = length(b - center);
    if (d > radius) { radius = d; }
    d = length(c - center);
    if (d > radius) { radius = d; }
}

}  // namespace detail

bool buildFromMesh(const Mesh &mesh, NaniteData &out)
{
    if (mesh.totvert == 0 || mesh.totface == 0) {
        return false;
    }

    out.header.magic = kNc2Magic;
    out.header.version = kNc2Version;
    out.header.vertexCount = static_cast<uint32_t>(mesh.totvert);
    out.header.indexCount = static_cast<uint32_t>(mesh.totface * 3);
    out.header.flags = 0;
    if (mesh.totloop > 0) { out.header.flags |= 1; }  // has UV
    if (mesh.rtvert > 0) { out.header.flags |= 2; }   // has normals
    out.header.boundsMin = float3(std::numeric_limits<float>::max());
    out.header.boundsMax = float3(-std::numeric_limits<float>::max());
    out.header.pad = 0;
    out.header.levelCount = 1;
    out.header.maxClusterTriangles = mesh.totface;
    out.header.maxClusterVertices = mesh.totvert;
    out.header.proxyVertexCount = mesh.totvert;
    out.header.proxyIndexCount = mesh.totface * 3;

    out.positions.reserve(mesh.totvert);
    out.indices.reserve(mesh.totface * 3);
    if (mesh.rtvert > 0) { out.normals.reserve(mesh.totvert); }
    if ((out.header.flags & 1) != 0 && mesh.totloop > 0) {
        out.uvs.reserve(mesh.totloop);
    }

    // Vertices
    for (int i = 0; i < mesh.totvert; ++i) {
        const MVert &v = mesh.verts[i];
        out.positions.push_back(float3(v.co.x, v.co.y, v.co.z));
        if (mesh.rtvert > 0) {
            out.normals.push_back(float3(v.norm[0], v.norm[1], v.norm[2]));
        }
    }
    // UVs
    if ((out.header.flags & 1) != 0 && mesh.totloop > 0) {
        for (int i = 0; i < mesh.totloop; ++i) {
            const MLoop &l = mesh.loops[i];
            out.uvs.push_back(float2(l.tx, l.ty));
        }
    }

    // Clusters
    out.header.clusterCount = static_cast<uint32_t>(mesh.totface);
    out.clusters.reserve(mesh.totface);
    out.proxyPositions = out.positions;
    out.proxyIndices.resize(out.indices.size());

    uint32_t nextIndex = 0;
    for (int i = 0; i < mesh.totface; ++i) {
        const Poly &p = mesh.polygons[i];
        const MVert *va = &mesh.verts[p.loopStart];
        const MVert *vb = &mesh.verts[p.loopStart + 1];
        const MVert *vc = &mesh.verts[p.loopStart + 2];

        float3 center = float3(0.0f);
        float radius = 0.0f;
        detail::computeTriangleSphere(float3(va->co), float3(vb->co),
                                      float3(vc->co), center, radius);

        NaniteCluster cluster;
        cluster.error = 0.0f;
        cluster.parentError = std::numeric_limits<float>::max();
        cluster.group = 0;
        cluster.parentGroup = -1;
        cluster.level = 0;
        cluster.firstIndex = nextIndex;
        cluster.indexCount = 3;
        cluster.sphere[0] = center.x;
        cluster.sphere[1] = center.y;
        cluster.sphere[2] = center.z;
        cluster.sphere[3] = radius;
        cluster.parentSphere[0] = center.x;
        cluster.parentSphere[1] = center.y;
        cluster.parentSphere[2] = center.z;
        cluster.parentSphere[3] = radius;
        cluster.flags = out.header.flags;

        out.proxyIndices[nextIndex + 0] = p.loopStart;
        out.proxyIndices[nextIndex + 1] = p.loopStart + 1;
        out.proxyIndices[nextIndex + 2] = p.loopStart + 2;
        out.clusters.push_back(cluster);

        out.indices.push_back(p.loopStart);
        out.indices.push_back(p.loopStart + 1);
        out.indices.push_back(p.loopStart + 2);
        nextIndex += 3;

        for (int k = 0; k < 3; ++k) {
            const float3 &pos = out.positions[p.loopStart + k];
            out.header.boundsMin.x = std::min(out.header.boundsMin.x, pos.x);
            out.header.boundsMin.y = std::min(out.header.boundsMin.y, pos.y);
            out.header.boundsMin.z = std::min(out.header.boundsMin.z, pos.z);
            out.header.boundsMax.x = std::max(out.header.boundsMax.x, pos.x);
            out.header.boundsMax.y = std::max(out.header.boundsMax.y, pos.y);
            out.header.boundsMax.z = std::max(out.header.boundsMax.z, pos.z);
        }
    }

    if (out.positions.empty()) {
        out.header.boundsMin = float3(0.0f);
        out.header.boundsMax = float3(0.0f);
    }
    return true;
}

}  // namespace nanite
}  // namespace blender
