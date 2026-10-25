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

bool extractMeshGeometry(const Mesh &mesh, TesseraRawMesh &out_raw)
{
    const Span<float3> vert_positions = mesh.vert_positions();
    const Span<int> corner_verts = mesh.corner_verts();
    const OffsetIndices<int> faces = mesh.faces();
    const Span<float3> vert_normals = mesh.vert_normals();

    /* Look up active UV layer: try active_uv_map_name(), then "UVMap", "uv_map", or any available UV map. */
    bke::AttributeReader<float2> uvs_reader;
    std::string uv_name = mesh.active_uv_map_name();
    if (!uv_name.empty()) {
        uvs_reader = mesh.attributes().lookup<float2>(uv_name, bke::AttrDomain::Corner);
    }
    if (!uvs_reader) {
        uvs_reader = mesh.attributes().lookup<float2>("UVMap", bke::AttrDomain::Corner);
    }
    if (!uvs_reader) {
        uvs_reader = mesh.attributes().lookup<float2>("uv_map", bke::AttrDomain::Corner);
    }
    if (!uvs_reader && !mesh.uv_map_names().is_empty()) {
        uvs_reader = mesh.attributes().lookup<float2>(mesh.uv_map_names()[0], bke::AttrDomain::Corner);
    }
    const bool has_uvs = bool(uvs_reader);

    const bke::AttributeReader<int> mat_reader = mesh.attributes().lookup<int>(
        "material_index", bke::AttrDomain::Face);
    const bool has_mat = bool(mat_reader);

    const int verts_num = vert_positions.size();
    const int faces_num = faces.size();
    if (verts_num == 0 || faces_num == 0) {
        return false;
    }

    out_raw.header_flags = 0;
    if (has_uvs) { out_raw.header_flags |= kNc2FlagHasUV; }
    if (!vert_normals.is_empty()) { out_raw.header_flags |= kNc2FlagHasNormals; }

    /* Input vertex pool with corner UV splitting so UV seams are preserved. */
    out_raw.verts.clear();
    out_raw.verts.reserve(verts_num);

    struct UVVertKey {
        float2 uv;
        uint32_t new_vert_id;
    };
    std::vector<std::vector<UVVertKey>> vert_to_corners(verts_num);

    auto get_or_create_vertex = [&](int orig_vert, int corner_idx) -> uint32_t {
        float2 uv = has_uvs ? uvs_reader.varray[corner_idx] : float2(0.0f);
        for (const auto &entry : vert_to_corners[orig_vert]) {
            if (std::abs(entry.uv.x - uv.x) < 1e-5f && std::abs(entry.uv.y - uv.y) < 1e-5f) {
                return entry.new_vert_id;
            }
        }
        TesseraRawVertex bv;
        bv.position = vert_positions[orig_vert];
        bv.normal = vert_normals.is_empty() ? float3(0.0f) : vert_normals[orig_vert];
        bv.uv = uv;
        bv.tangent = float4(1.0f, 0.0f, 0.0f, 1.0f);
        uint32_t new_id = uint32_t(out_raw.verts.size());
        out_raw.verts.push_back(bv);
        vert_to_corners[orig_vert].push_back({uv, new_id});
        return new_id;
    };

    /* Input triangles: fan triangulation of every face with material extraction. */
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
    out_raw.tri_verts.clear();
    out_raw.tri_verts.reserve(tri_reserve * 3);
    out_raw.tri_materials.clear();
    out_raw.tri_materials.reserve(tri_reserve);

    for (const int i : faces.index_range()) {
        const IndexRange face = faces[i];
        if (face.size() < 3) {
            continue;
        }
        int mat_idx = has_mat ? mat_reader.varray[i] : 0;
        if (mat_idx < 0) {
            mat_idx = 0;
        }
        const int c0 = face.start();
        const uint32_t v0 = get_or_create_vertex(corner_verts[c0], c0);
        for (int k = 1; k + 1 < face.size(); ++k) {
            const int c1 = face.start() + k;
            const int c2 = face.start() + k + 1;
            out_raw.tri_verts.push_back(v0);
            out_raw.tri_verts.push_back(get_or_create_vertex(corner_verts[c1], c1));
            out_raw.tri_verts.push_back(get_or_create_vertex(corner_verts[c2], c2));
            out_raw.tri_materials.push_back(uint32_t(mat_idx));
        }
    }

    /* Compute 4D MikkTSpace-compatible tangents if UV coordinates exist. */
    if (has_uvs) {
        const int build_verts_num = int(out_raw.verts.size());
        std::vector<float3> tan_accum(build_verts_num, float3(0.0f));
        std::vector<float3> bitan_accum(build_verts_num, float3(0.0f));
        for (size_t t = 0; t + 2 < out_raw.tri_verts.size(); t += 3) {
            uint32_t i0 = out_raw.tri_verts[t];
            uint32_t i1 = out_raw.tri_verts[t + 1];
            uint32_t i2 = out_raw.tri_verts[t + 2];
            const float3 &p0 = out_raw.verts[i0].position;
            const float3 &p1 = out_raw.verts[i1].position;
            const float3 &p2 = out_raw.verts[i2].position;
            const float2 &uv0 = out_raw.verts[i0].uv;
            const float2 &uv1 = out_raw.verts[i1].uv;
            const float2 &uv2 = out_raw.verts[i2].uv;
            float3 e1 = p1 - p0;
            float3 e2 = p2 - p0;
            float du1 = uv1.x - uv0.x;
            float dv1 = uv1.y - uv0.y;
            float du2 = uv2.x - uv0.x;
            float dv2 = uv2.y - uv0.y;
            float det = du1 * dv2 - du2 * dv1;
            if (std::abs(det) > 1e-7f) {
                float r = 1.0f / det;
                float3 t_vec = (e1 * dv2 - e2 * dv1) * r;
                float3 b_vec = (e2 * du1 - e1 * du2) * r;
                tan_accum[i0] += t_vec;
                tan_accum[i1] += t_vec;
                tan_accum[i2] += t_vec;
                bitan_accum[i0] += b_vec;
                bitan_accum[i1] += b_vec;
                bitan_accum[i2] += b_vec;
            }
        }
        for (int i = 0; i < build_verts_num; ++i) {
            const float3 &n = out_raw.verts[i].normal;
            const float3 &t_raw = tan_accum[i];
            const float3 &b_raw = bitan_accum[i];
            float3 t_ortho = t_raw - n * math::dot(n, t_raw);
            float len = math::length(t_ortho);
            if (len > 1e-6f) {
                t_ortho /= len;
            } else {
                t_ortho = float3(1.0f, 0.0f, 0.0f);
            }
            float sign = (math::dot(math::cross(n, t_ortho), b_raw) < 0.0f) ? -1.0f : 1.0f;
            out_raw.verts[i].tangent = float4(t_ortho.x, t_ortho.y, t_ortho.z, sign);
        }
        out_raw.header_flags |= kNc2FlagHasTangents;
    }

    return true;
}

bool buildFromRawMesh(const TesseraRawMesh &raw, TesseraData &out, TesseraProgressCallback progress_cb)
{
    if (raw.verts.empty() || raw.tri_verts.empty()) {
        return false;
    }

    out.header.magic = kNc2Magic;
    out.header.version = kNc2Version;
    out.header.flags = raw.header_flags;

    /* Build the cluster DAG: region-growing clustering plus hierarchical
     * merge + QEM simplification, partitioned strictly per material. */
    detail::BuilderParams params;
    const detail::BuildResult dag = detail::buildClusterDAG(
        raw.verts, raw.tri_verts, params, raw.tri_materials, progress_cb);
    if (dag.clusters.empty()) {
        return false;
    }

    if (progress_cb) {
        progress_cb(0.90f, "Finalizing Tessera clusters & proxy data...");
    }

    /* Per-vertex attribute buffers (all clusters share one vertex pool). */
    out.positions.resize(dag.verts.size());
    for (size_t i = 0; i < dag.verts.size(); ++i) {
        out.positions[i] = dag.verts[i].position;
    }
    const bool has_normals = (out.header.flags & kNc2FlagHasNormals) != 0;
    if (has_normals) {
        out.normals.resize(dag.verts.size());
        for (size_t i = 0; i < dag.verts.size(); ++i) {
            out.normals[i] = dag.verts[i].normal;
        }
    }
    else {
        out.normals.clear();
    }
    const bool has_uvs_flag = (out.header.flags & kNc2FlagHasUV) != 0;
    if (has_uvs_flag) {
        out.uvs.resize(dag.verts.size());
        for (size_t i = 0; i < dag.verts.size(); ++i) {
            out.uvs[i] = dag.verts[i].uv;
        }
    }
    else {
        out.uvs.clear();
    }
    const bool has_tangents_flag = (out.header.flags & kNc2FlagHasTangents) != 0;
    if (has_tangents_flag) {
        out.tangents.resize(dag.verts.size());
        for (size_t i = 0; i < dag.verts.size(); ++i) {
            out.tangents[i] = dag.verts[i].tangent;
        }
    }
    else {
        out.tangents.clear();
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
        nc.flags = uint16_t(out.header.flags);
        nc.materialIndex = uint16_t(c.material_index);

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

bool buildFromMesh(const Mesh &mesh, TesseraData &out, TesseraProgressCallback progress_cb)
{
    if (progress_cb) {
        progress_cb(0.0f, "Preparing mesh vertices and UVs...");
    }
    TesseraRawMesh raw;
    if (!extractMeshGeometry(mesh, raw)) {
        return false;
    }
    return buildFromRawMesh(raw, out, progress_cb);
}

}  // namespace tessera
}  // namespace blender
