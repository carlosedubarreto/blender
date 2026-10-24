/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Cluster builder and serialization implementation.
 */

#include "tessera_cluster.hh"

#include "BKE_attribute.hh"
#include "BKE_mesh.hh"

#include "BLI_index_range.hh"
#include "BLI_math_vector.hh"

#include "tessera_cluster_builder.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

namespace blender {
namespace tessera {
namespace detail {

void computeTriangleSphere(const float3 &a, const float3 &b, const float3 &c,
                           float3 &center, float &radius)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float dAB = math::length_squared(ab);
    const float dAC = math::length_squared(ac);
    const float dBC = math::length_squared(bc);
    const float3 e1 = (dAB >= dAC) ? ((dAB >= dBC) ? ab : bc)
                                   : ((dAC >= dBC) ? ac : bc);
    const float3 e2 = (dAB >= dAC) ? ((dAB >= dBC) ? ac : bc)
                                   : ((dAC >= dBC) ? ab : bc);
    const float3 mid = (e1 + e2) * 0.5f;
    float len = math::length(e2 - e1);
    len = len > 0.0f ? len : 1.0f;
    center = mid;
    radius = math::length(e2 - mid);
    float d = math::distance(a, center);
    if (d > radius) { radius = d; }
    d = math::distance(b, center);
    if (d > radius) { radius = d; }
    d = math::distance(c, center);
    if (d > radius) { radius = d; }
}

}  // namespace detail

bool buildFromMesh(const Mesh &mesh, TesseraData &out)
{
    const Span<float3> vert_positions = mesh.vert_positions();
    const Span<int> corner_verts = mesh.corner_verts();
    const OffsetIndices<int> faces = mesh.faces();
    const Span<float3> vert_normals = mesh.vert_normals();
    const bke::AttributeReader<float2> uvs_reader = mesh.attributes().lookup<float2>(
        "uv_map", bke::AttrDomain::Corner);
    const bool has_uvs = bool(uvs_reader);

    const int verts_num = vert_positions.size();
    const int faces_num = faces.size();
    if (verts_num == 0 || faces_num == 0) {
        return false;
    }

    out.header.magic = kNc2Magic;
    out.header.version = kNc2Version;
    out.header.flags = 0;
    if (has_uvs) { out.header.flags |= 1; }                   // has UV
    if (!vert_normals.is_empty()) { out.header.flags |= 2; }  // has normals

    /* Input vertex pool with attributes. */
    std::vector<detail::BuildVertex> input_verts;
    input_verts.resize(size_t(verts_num));
    for (const int i : IndexRange(verts_num)) {
        input_verts[i].position = vert_positions[i];
        input_verts[i].normal = vert_normals.is_empty() ? float3(0.0f) :
                                                          vert_normals[i];
        input_verts[i].uv = has_uvs ? uvs_reader.varray[i] : float2(0.0f);
    }

    /* Input triangles: fan triangulation of every face. */
    size_t tri_reserve = 0;
    for (const int i : faces.index_range()) {
        const int face_size = faces[i].size();
        if (face_size >= 3) {
            tri_reserve += face_size - 2;
        }
    }
    if (tri_reserve == 0) {
        return false;
    }
    std::vector<uint32_t> tri_verts;
    tri_verts.reserve(tri_reserve * 3);
    for (const int i : faces.index_range()) {
        const IndexRange face = faces[i];
        if (face.size() < 3) {
            continue;
        }
        const int v0 = corner_verts[face.start()];
        for (int k = 1; k + 1 < face.size(); ++k) {
            tri_verts.push_back(uint32_t(v0));
            tri_verts.push_back(uint32_t(corner_verts[face.start() + k]));
            tri_verts.push_back(uint32_t(corner_verts[face.start() + k + 1]));
        }
    }

    /* Build the cluster DAG: region-growing clustering plus hierarchical
     * merge + QEM simplification. This is the core Tessera front end. */
    detail::BuilderParams params;
    const detail::BuildResult dag = detail::buildClusterDAG(
        input_verts, tri_verts, params);
    if (dag.clusters.empty()) {
        return false;
    }

    /* Per-vertex attribute buffers (all clusters share one vertex pool). */
    out.positions.resize(dag.verts.size());
    for (size_t i = 0; i < dag.verts.size(); ++i) {
        out.positions[i] = dag.verts[i].position;
    }
    const bool has_normals = (out.header.flags & 2) != 0;
    if (has_normals) {
        out.normals.resize(dag.verts.size());
        for (size_t i = 0; i < dag.verts.size(); ++i) {
            out.normals[i] = dag.verts[i].normal;
        }
    }
    else {
        out.normals.clear();
    }
    const bool has_uvs_flag = (out.header.flags & 1) != 0;
    if (has_uvs_flag) {
        out.uvs.resize(dag.verts.size());
        for (size_t i = 0; i < dag.verts.size(); ++i) {
            out.uvs[i] = dag.verts[i].uv;
        }
    }
    else {
        out.uvs.clear();
    }

    // Clusters: assign contiguous index-buffer ranges and convert DAG nodes.
    out.header.clusterCount = static_cast<uint32_t>(dag.clusters.size());
    out.clusters.reserve(dag.clusters.size());
    out.indices.reserve(dag.tris.size() * 3);
    uint32_t max_cluster_tris = 0;
    uint32_t max_cluster_verts = 0;

    for (const auto &c : dag.clusters) {
        const uint32_t first_index = uint32_t(out.indices.size());
        for (const uint32_t t : c.tris) {
            out.indices.push_back(dag.tris[t].v[0]);
            out.indices.push_back(dag.tris[t].v[1]);
            out.indices.push_back(dag.tris[t].v[2]);
        }

        TesseraCluster nc = {};
        nc.error = c.error;
        nc.parentError = c.parentError;
        nc.firstIndex = first_index;
        nc.indexCount = uint32_t(c.tris.size() * 3);
        nc.parentGroup = -1;
        nc.group = uint32_t(out.clusters.size());
        nc.level = c.level;
        nc.sphere[0] = c.sphereCenter.x;
        nc.sphere[1] = c.sphereCenter.y;
        nc.sphere[2] = c.sphereCenter.z;
        nc.sphere[3] = c.sphereRadius;
        nc.parentSphere[0] = c.sphereCenter.x;
        nc.parentSphere[1] = c.sphereCenter.y;
        nc.parentSphere[2] = c.sphereCenter.z;
        nc.parentSphere[3] = c.sphereRadius;
        nc.flags = out.header.flags;

        /* Unique vertices referenced by this cluster. */
        std::unordered_set<uint32_t> unique_verts;
        for (const uint32_t t : c.tris) {
            for (int k = 0; k < 3; ++k) {
                unique_verts.insert(dag.tris[t].v[k]);
            }
        }
        max_cluster_tris = std::max(max_cluster_tris, uint32_t(c.tris.size()));
        max_cluster_verts = std::max(max_cluster_verts,
                                     uint32_t(unique_verts.size()));

        out.clusters.push_back(nc);
    }

    /* Parent links (DAG edges). */
    for (size_t ci = 0; ci < dag.clusters.size(); ++ci) {
        const auto &c = dag.clusters[ci];
        for (const uint32_t child : c.children) {
            out.clusters[child].parentGroup = int32_t(ci);
            out.clusters[child].parentSphere[0] = c.sphereCenter.x;
            out.clusters[child].parentSphere[1] = c.sphereCenter.y;
            out.clusters[child].parentSphere[2] = c.sphereCenter.z;
            out.clusters[child].parentSphere[3] = c.sphereRadius;
        }
    }

    out.header.vertexCount = uint32_t(dag.verts.size());
    out.header.indexCount = uint32_t(out.indices.size());
    out.header.levelCount = dag.levelCount;
    out.header.maxClusterTriangles = max_cluster_tris;
    out.header.maxClusterVertices = max_cluster_verts;

    /* Proxy: the finest (level 0) geometry, used for simple non-Tessera
     * rendering and for occlusion proxying. */
    out.proxyPositions = out.positions;
    out.proxyIndices.clear();
    uint32_t proxy_tris = 0;
    for (const auto &nc : out.clusters) {
        if (nc.level != 0) {
            continue;
        }
        proxy_tris += nc.indexCount / 3;
        for (uint32_t k = 0; k < nc.indexCount; ++k) {
            out.proxyIndices.push_back(out.indices[nc.firstIndex + k]);
        }
    }
    out.header.proxyVertexCount = uint32_t(dag.verts.size());
    out.header.proxyIndexCount = uint32_t(out.proxyIndices.size());

    /* Total bounds from the finest geometry. */
    out.header.boundsMin[0] = out.header.boundsMin[1] = out.header.boundsMin[2] =
        std::numeric_limits<float>::max();
    out.header.boundsMax[0] = out.header.boundsMax[1] = out.header.boundsMax[2] =
        -std::numeric_limits<float>::max();
    for (const uint32_t idx : out.proxyIndices) {
        const float3 &pos = out.positions[idx];
        out.header.boundsMin[0] = std::min(out.header.boundsMin[0], pos.x);
        out.header.boundsMin[1] = std::min(out.header.boundsMin[1], pos.y);
        out.header.boundsMin[2] = std::min(out.header.boundsMin[2], pos.z);
        out.header.boundsMax[0] = std::max(out.header.boundsMax[0], pos.x);
        out.header.boundsMax[1] = std::max(out.header.boundsMax[1], pos.y);
        out.header.boundsMax[2] = std::max(out.header.boundsMax[2], pos.z);
    }
    if (out.positions.empty()) {
        out.header.boundsMin[0] = out.header.boundsMin[1] = out.header.boundsMin[2] = 0.0f;
        out.header.boundsMax[0] = out.header.boundsMax[1] = out.header.boundsMax[2] = 0.0f;
    }
    return true;
}

}  // namespace tessera
}  // namespace blender
