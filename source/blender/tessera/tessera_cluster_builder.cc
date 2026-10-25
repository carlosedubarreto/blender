/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Cluster DAG builder implementation.
 */

#include "tessera_cluster_builder.hh"

#include "tessera_cluster.hh"
#include "tessera_qem.hh"

#include "BLI_math_vector.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace blender {
namespace tessera {
namespace detail {

namespace {

constexpr uint32_t kInvalid = 0xffffffffu;

/* 64-bit key for an exactly-coinciding vertex position (bit pattern hash).
 * Cluster-private copies of shared vertices are bit-identical, so this
 * detects true geometric adjacency even across cluster boundaries. */
uint64_t position_key(const float3 &p)
{
    uint32_t b[3];
    std::memcpy(b, &p.x, sizeof(b));
    uint64_t h = uint64_t(b[0]) * 0x9E3779B97F4A7C15ULL;
    h ^= uint64_t(b[1]) * 0xC2B2AE3D27D4EB4FULL;
    h ^= uint64_t(b[2]) * 0x165667B19E3779F9ULL;
    h ^= h >> 29;
    return h;
}

uint64_t edge_key(uint32_t a, uint32_t b)
{
    if (a > b) {
        std::swap(a, b);
    }
    return (static_cast<uint64_t>(a) << 32) | b;
}

/* Map of edge -> triangles using it, within a triangle subset. */
using EdgeMap = std::unordered_map<uint64_t, std::vector<uint32_t>>;

void build_edge_map(const std::vector<BuildTriangle> &tris,
                    const std::vector<uint32_t> &subset,
                    EdgeMap &edges)
{
    edges.clear();
    for (const uint32_t t : subset) {
        const auto &tv = tris[t].v;
        if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            edges[edge_key(tv[k], tv[(k + 1) % 3])].push_back(t);
        }
    }
}

/**
 * Partition a triangle subset into clusters of at most \a target triangles
 * using region growing: each cluster expands across the cheapest shared edge
 * (flat continuations first) until the target size is reached.
 */
std::vector<std::vector<uint32_t>> partitionTriangles(
    const std::vector<BuildTriangle> &tris,
    const std::vector<uint32_t> &subset,
    const std::vector<float3> &positions,
    uint32_t target)
{
    std::vector<std::vector<uint32_t>> clusters;
    if (subset.empty() || target == 0) {
        if (!subset.empty()) {
            clusters.push_back(subset);
        }
        return clusters;
    }

    EdgeMap edges;
    build_edge_map(tris, subset, edges);

    std::unordered_map<uint64_t, float3> edge_normal;
    std::vector<bool> visited(tris.size(), false);
    for (const uint32_t t : subset) {
        const auto &tv = tris[t].v;
        if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
            visited[t] = true;  /* degenerate, never clustered */
        }
    }

    struct Seed {
        float cost;
        uint32_t tri;
        uint32_t from;  /* tri the candidate was discovered from */
        bool operator>(const Seed &o) const { return cost > o.cost; }
    };

    /* Membership stamp: cluster_stamp[t] == current cluster id + 1 when the
     * triangle belongs to the cluster being grown. */
    std::vector<uint32_t> cluster_stamp(tris.size(), 0);
    constexpr float kBoundaryWeight = 4.0f;

    for (const uint32_t t : subset) {
        if (visited[t]) {
            continue;
        }
        clusters.emplace_back();
        std::vector<uint32_t> &cluster = clusters.back();
        const uint32_t stamp = uint32_t(clusters.size());
        std::priority_queue<Seed, std::vector<Seed>, std::greater<Seed>> pq;
        visited[t] = true;
        cluster_stamp[t] = stamp;
        cluster.push_back(t);
        float3 centroid_sum = positions[tris[t].v[0]] +
                              positions[tris[t].v[1]] +
                              positions[tris[t].v[2]];
        /* Expand until the target size is reached. The cost combines the
         * dihedral angle (flat continuations first) with the number of NEW
         * boundary edges the candidate would add — this keeps clusters
         * compact (disks), which is essential: strip/ring shaped clusters
         * have no interior vertices and cannot be simplified. */
        auto push_neighbors = [&](uint32_t tri) {
            const auto &tv = tris[tri].v;
            for (int k = 0; k < 3; ++k) {
                const auto it = edges.find(edge_key(tv[k], tv[(k + 1) % 3]));
                if (it == edges.end()) {
                    continue;
                }
                for (const uint32_t other : it->second) {
                    if (visited[other] || other == tri) {
                        continue;
                    }
                    const auto &tn = tris[other].v;
                    const float3 n0 = math::normalize(
                        math::cross(positions[tv[1]] - positions[tv[0]],
                                    positions[tv[2]] - positions[tv[0]]));
                    const float3 n1 = math::normalize(
                        math::cross(positions[tn[1]] - positions[tn[0]],
                                    positions[tn[2]] - positions[tn[0]]));
                    const uint64_t shared_key = edge_key(tv[k],
                                                         tv[(k + 1) % 3]);
                    float new_boundary = 0.0f;
                    for (int e = 0; e < 3; ++e) {
                        const uint64_t ekey = edge_key(tn[e],
                                                       tn[(e + 1) % 3]);
                        if (ekey == shared_key) {
                            continue;  /* the shared edge is interior */
                        }
                        const auto it2 = edges.find(
                            edge_key(tn[e], tn[(e + 1) % 3]));
                        bool interior = false;
                        if (it2 != edges.end()) {
                            for (const uint32_t nb_tri : it2->second) {
                                if (nb_tri != other &&
                                    cluster_stamp[nb_tri] == stamp) {
                                    interior = true;
                                    break;
                                }
                            }
                        }
                        if (!interior) {
                            new_boundary += 1.0f;
                        }
                    }
                    const float3 cand_centroid = (positions[tn[0]] +
                                                  positions[tn[1]] +
                                                  positions[tn[2]]) *
                                                 (1.0f / 3.0f);
                    const float3 cluster_centroid = centroid_sum *
                                                    (1.0f /
                                                     float(3 * cluster.size()));
                    const float cost = kBoundaryWeight * new_boundary +
                                       (1.0f - math::dot(n0, n1)) +
                                       0.2f * math::distance(cand_centroid,
                                                             cluster_centroid);
                    pq.push({cost, other, tri});
                }
            }
        };
        push_neighbors(t);
        while (cluster.size() < target && !pq.empty()) {
            const Seed s = pq.top();
            pq.pop();
            if (visited[s.tri]) {
                continue;
            }
            visited[s.tri] = true;
            cluster.push_back(s.tri);
            centroid_sum += positions[tris[s.tri].v[0]] +
                            positions[tris[s.tri].v[1]] +
                            positions[tris[s.tri].v[2]];
            push_neighbors(s.tri);
        }
    }
    /* Absorb tiny fragment clusters (QEM can disconnect a patch) into their
     * nearest neighbor: parent clusters made of fragments would multiply the
     * frontier instead of shrinking it. Iterates smallest-first so groups of
     * only-small parts also coalesce into one cluster. */
    const uint32_t min_size = std::max(uint32_t(1), target / 4);
    auto compute_centers = [&]() {
        std::vector<float3> centers(clusters.size());
        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            float3 sum(0.0f);
            for (const uint32_t t : clusters[ci]) {
                sum += positions[tris[t].v[0]];
                sum += positions[tris[t].v[1]];
                sum += positions[tris[t].v[2]];
            }
            centers[ci] = clusters[ci].empty()
                              ? float3(0.0f)
                              : sum / float(3 * clusters[ci].size());
        }
        return centers;
    };
    for (;;) {
        /* Find the smallest non-empty cluster. */
        size_t si = clusters.size();
        size_t smallest = SIZE_MAX;
        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            if (!clusters[ci].empty() && clusters[ci].size() < smallest) {
                smallest = clusters[ci].size();
                si = ci;
            }
        }
        if (si == clusters.size() || smallest >= min_size ||
            clusters.size() <= 1)
        {
            break;
        }
        /* Absorb it into the nearest other cluster. */
        const std::vector<float3> centers = compute_centers();
        size_t best = clusters.size();
        float best_d = std::numeric_limits<float>::max();
        for (size_t cj = 0; cj < clusters.size(); ++cj) {
            if (cj == si || clusters[cj].empty()) {
                continue;
            }
            const float d = math::distance_squared(centers[si], centers[cj]);
            if (d < best_d) {
                best_d = d;
                best = cj;
            }
        }
        if (best == clusters.size()) {
            break;
        }
        clusters[best].insert(clusters[best].end(),
                              clusters[si].begin(),
                              clusters[si].end());
        clusters[si].clear();
    }
    /* Drop absorbed entries. */
    std::vector<std::vector<uint32_t>> compacted;
    compacted.reserve(clusters.size());
    for (auto &c : clusters) {
        if (!c.empty()) {
            compacted.push_back(std::move(c));
        }
    }
    clusters = std::move(compacted);
    return clusters;
}

// --- PART2 ---

/** A self-contained triangle set in local vertex space. */
struct LocalMesh {
    std::vector<BuildVertex> verts;
    std::vector<uint32_t> tris;  /* flat, 3 local ids per triangle */
};

void extract_positions(const std::vector<BuildVertex> &verts,
                       std::vector<float3> &out)
{
    out.resize(verts.size());
    for (size_t i = 0; i < verts.size(); ++i) {
        out[i] = verts[i].position;
    }
}

void local_to_tri_array(const LocalMesh &local,
                        std::vector<BuildTriangle> &out)
{
    out.resize(local.tris.size() / 3);
    for (size_t t = 0; t < out.size(); ++t) {
        out[t].v[0] = local.tris[t * 3 + 0];
        out[t].v[1] = local.tris[t * 3 + 1];
        out[t].v[2] = local.tris[t * 3 + 2];
    }
}

/**
 * Create a cluster from a subset of local triangles: appends private vertex
 * copies (deduplicated within the cluster) and its triangles to the result
 * pool, computes the bounding sphere, and returns the new cluster id.
 */
uint32_t makeCluster(BuildResult &R,
                     const LocalMesh &local,
                     const std::vector<uint32_t> &local_tri_ids,
                     float error,
                     float parent_error,
                     uint32_t level,
                     bool copy_verts)
{
    BuildCluster c;
    c.error = error;
    c.parentError = parent_error;
    c.level = level;

    std::unordered_map<uint32_t, uint32_t> local_to_pool;
    std::vector<uint32_t> tri_ids;
    tri_ids.reserve(local_tri_ids.size());
    float3 bmin(std::numeric_limits<float>::max());
    float3 bmax(-std::numeric_limits<float>::max());

    auto track_pos = [&](const float3 &p) {
        bmin = math::min(bmin, p);
        bmax = math::max(bmax, p);
    };

    if (!copy_verts) {
        /* The local mesh is the input mesh itself: triangle ids index into
         * R.tris directly and vertices stay shared (needed so cluster
         * adjacency by position/vertex works at the finest level). */
        for (const uint32_t lt : local_tri_ids) {
            const auto &tv = R.tris[lt].v;
            const float3 &p0 = R.verts[tv[0]].position;
            const float3 &p1 = R.verts[tv[1]].position;
            const float3 &p2 = R.verts[tv[2]].position;
            if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2] ||
                math::length_squared(math::cross(p1 - p0, p2 - p0)) <=
                    1e-20f)
            {
                continue;  /* degenerate: nothing to render */
            }
            tri_ids.push_back(lt);
            track_pos(p0);
            track_pos(p1);
            track_pos(p2);
        }
    }
    else {
        for (const uint32_t lt : local_tri_ids) {
            BuildTriangle nt;
            bool degenerate = false;
            for (int k = 0; k < 3; ++k) {
                const uint32_t lv = local.tris[lt * 3 + k];
                auto it = local_to_pool.find(lv);
                if (it == local_to_pool.end()) {
                    const uint32_t pool_id = uint32_t(R.verts.size());
                    R.verts.push_back(local.verts[lv]);
                    local_to_pool.emplace(lv, pool_id);
                    it = local_to_pool.find(lv);
                }
                nt.v[k] = it->second;
                track_pos(local.verts[lv].position);
            }
            {
                const float3 &p0 = R.verts[nt.v[0]].position;
                const float3 &p1 = R.verts[nt.v[1]].position;
                const float3 &p2 = R.verts[nt.v[2]].position;
                degenerate = nt.v[0] == nt.v[1] || nt.v[1] == nt.v[2] ||
                             nt.v[0] == nt.v[2] ||
                             math::length_squared(math::cross(p1 - p0,
                                                              p2 - p0)) <=
                                 1e-20f;
            }
            if (degenerate) {
                continue;  /* zero-area debris: skip */
            }
            tri_ids.push_back(uint32_t(R.tris.size()));
            R.tris.push_back(nt);
        }
    }

    c.tris = std::move(tri_ids);
    c.sphereCenter = (bmin + bmax) * 0.5f;
    c.sphereRadius = 0.0f;
    for (const uint32_t t : c.tris) {
        for (int k = 0; k < 3; ++k) {
            c.sphereRadius = std::max(
                c.sphereRadius,
                math::distance(R.verts[R.tris[t].v[k]].position,
                               c.sphereCenter));
        }
    }
    const uint32_t id = uint32_t(R.clusters.size());
    R.clusters.push_back(std::move(c));
    return id;
}

/** Concatenate the geometry of the given clusters into one local mesh.
 * Vertices coinciding exactly (bit-identical positions) are unified, so the
 * QEM pass sees true interior edges across cluster boundaries. */
void mergedLocalFromClusters(const BuildResult &R,
                             const std::vector<uint32_t> &cluster_ids,
                             LocalMesh &out)
{
    std::unordered_map<uint64_t, uint32_t> pos_to_local;
    for (const uint32_t cid : cluster_ids) {
        for (const uint32_t t : R.clusters[cid].tris) {
            uint32_t lv[3];
            for (int k = 0; k < 3; ++k) {
                const uint32_t pv = R.tris[t].v[k];
                const uint64_t key = position_key(R.verts[pv].position);
                auto it = pos_to_local.find(key);
                if (it == pos_to_local.end()) {
                    const uint32_t local_id = uint32_t(out.verts.size());
                    out.verts.push_back(R.verts[pv]);
                    pos_to_local.emplace(key, local_id);
                    it = pos_to_local.find(key);
                }
                lv[k] = it->second;
            }
            out.tris.push_back(lv[0]);
            out.tris.push_back(lv[1]);
            out.tris.push_back(lv[2]);
        }
    }
}

/**
 * Group the frontier clusters into merge groups of at most \a max_size
 * adjacent clusters (adjacency = sharing a vertex). Greedy BFS keeps spatial
 * locality, which keeps parent clusters compact.
 */
std::vector<std::vector<uint32_t>> groupClusters(const BuildResult &R,
                                                 const std::vector<uint32_t> &frontier,
                                                 uint32_t max_size)
{
    /* Cluster adjacency via coinciding vertex positions (cluster-private
     * copies of shared vertices are bit-identical). */
    std::unordered_map<uint64_t, std::vector<uint32_t>> pos_clusters;
    for (const uint32_t cid : frontier) {
        for (const uint32_t t : R.clusters[cid].tris) {
            for (int k = 0; k < 3; ++k) {
                pos_clusters[position_key(
                                 R.verts[R.tris[t].v[k]].position)]
                    .push_back(cid);
            }
        }
    }
    std::unordered_map<uint32_t, std::vector<uint32_t>> adjacency;
    for (const auto &pc : pos_clusters) {
        const auto &list = pc.second;
        for (size_t i = 0; i + 1 < list.size(); ++i) {
            for (size_t j = i + 1; j < list.size(); ++j) {
                adjacency[list[i]].push_back(list[j]);
                adjacency[list[j]].push_back(list[i]);
            }
        }
    }

    /* Greedy BFS grouping. Clusters are only marked grouped when actually
     * added to a group; queued candidates stay available as seeds, so no
     * cluster is ever orphaned. */
    std::unordered_set<uint32_t> grouped;
    std::vector<std::vector<uint32_t>> groups;
    for (const uint32_t cid : frontier) {
        if (grouped.count(cid)) {
            continue;
        }
        groups.emplace_back();
        std::vector<uint32_t> &group = groups.back();
        std::vector<uint32_t> queue{cid};
        std::unordered_set<uint32_t> seen{cid};
        while (!queue.empty() && group.size() < max_size) {
            const uint32_t cur = queue.back();
            queue.pop_back();
            grouped.insert(cur);
            group.push_back(cur);
            if (group.size() == max_size) {
                break;
            }
            for (const uint32_t nb : adjacency[cur]) {
                if (R.clusters[nb].material_index != R.clusters[cur].material_index) {
                    continue;
                }
                if (!seen.count(nb) && !grouped.count(nb)) {
                    seen.insert(nb);
                    queue.push_back(nb);
                }
            }
        }
    }
    return groups;
}

// --- PART4 ---

}  // namespace

BuildResult buildClusterDAG(const std::vector<BuildVertex> &verts,
                            const std::vector<uint32_t> &tri_verts,
                            const BuilderParams &params,
                            const std::vector<uint32_t> &tri_materials,
                            TesseraProgressCallback progress_cb)
{
    BuildResult R;
    R.verts = verts;
    R.tris.reserve(tri_verts.size() / 3);
    for (size_t i = 0; i + 2 < tri_verts.size(); i += 3) {
        const size_t tri_idx = i / 3;
        const uint32_t mat_id = (tri_idx < tri_materials.size()) ? tri_materials[tri_idx] : 0u;
        R.tris.push_back({{tri_verts[i], tri_verts[i + 1], tri_verts[i + 2]}, mat_id});
    }
    if (R.tris.empty()) {
        return R;
    }

    if (progress_cb) {
        progress_cb(0.05f, "Partitioning Level 0 clusters...");
    }

    /* Level 0: partition the input mesh into fine clusters partitioned per material. */
    const std::vector<BuildTriangle> tri_array = R.tris;
    std::vector<float3> positions;
    extract_positions(R.verts, positions);

    std::unordered_map<uint32_t, std::vector<uint32_t>> mat_tris;
    for (uint32_t i = 0; i < uint32_t(tri_array.size()); ++i) {
        mat_tris[tri_array[i].material_index].push_back(i);
    }

    std::vector<uint32_t> frontier;
    LocalMesh input_local;
    input_local.verts = R.verts;
    input_local.tris = tri_verts;

    size_t mat_count = mat_tris.size();
    size_t mat_processed = 0;
    for (const auto &pair : mat_tris) {
        const uint32_t mat_id = pair.first;
        const auto &subset = pair.second;
        const std::vector<std::vector<uint32_t>> level0_for_mat = partitionTriangles(
            tri_array, subset, positions, params.maxClusterTris);

        for (const auto &part : level0_for_mat) {
            if (part.empty()) {
                continue;
            }
            /* Level 0 clusters reference the input vertices directly (error 0). */
            const uint32_t id = makeCluster(
                R, input_local, part, 0.0f,
                std::numeric_limits<float>::max(), 0, /*copy_verts=*/false);
            R.clusters[id].material_index = mat_id;
            frontier.push_back(id);
        }
        mat_processed++;
        if (progress_cb) {
            float p = 0.05f + 0.25f * (float(mat_processed) / float(mat_count));
            progress_cb(p, "Partitioning Level 0 clusters...");
        }
    }
    R.levelCount = 1;
    if (frontier.size() <= 1) {
        if (progress_cb) {
            progress_cb(0.85f, "Level 0 complete (single cluster).");
        }
        return R;  /* small mesh: single root cluster */
    }

    /* Hierarchy: merge adjacent clusters, simplify with QEM, re-cluster. */
    for (uint32_t level = 1; level < params.maxLevels; ++level) {
        if (progress_cb) {
            float p = 0.30f + 0.55f * std::min(1.0f, float(level) / 16.0f);
            char step_msg[64];
            snprintf(step_msg, sizeof(step_msg), "Simplifying DAG Level %u...", level);
            progress_cb(p, step_msg);
        }
        const auto groups = groupClusters(R, frontier, params.mergeGroupSize);
        std::vector<uint32_t> next_frontier;
        next_frontier.reserve(groups.size());

        for (const auto &mg : groups) {
            if (mg.empty()) {
                continue;
            }
            const uint32_t group_mat = R.clusters[mg.front()].material_index;
            LocalMesh merged;
            mergedLocalFromClusters(R, mg, merged);
            const uint32_t merged_tris = uint32_t(merged.tris.size() / 3);
            if (merged_tris < 4) {
                continue;
            }

            /* Simplify the merged patch to (about) half its triangles. */
            const uint32_t target = std::min(
                params.maxClusterTris,
                std::max(uint32_t(1),
                         uint32_t(std::ceil(float(merged_tris) *
                                            params.simplifyRatio))));
            std::vector<float3> merged_positions;
            extract_positions(merged.verts, merged_positions);
            const SimplifyResult simplified = simplifyTriangles(
                merged_positions, merged.tris, target);
            if (simplified.tris.size() < 9) {
                continue;  /* nothing left after simplification */
            }

            LocalMesh coarse;
            coarse.tris = simplified.tris;
            coarse.verts.resize(simplified.positions.size());
            for (size_t i = 0; i < simplified.positions.size(); ++i) {
                coarse.verts[i].position = simplified.positions[i];
                if (i < merged.verts.size()) {
                    coarse.verts[i].normal = merged.verts[i].normal;
                    coarse.verts[i].uv = merged.verts[i].uv;
                    coarse.verts[i].tangent = merged.verts[i].tangent;
                }
            }

            /* Re-cluster the simplified patch into parent clusters. */
            std::vector<BuildTriangle> coarse_tri_array;
            local_to_tri_array(coarse, coarse_tri_array);
            std::vector<uint32_t> coarse_all(coarse_tri_array.size());
            for (uint32_t i = 0; i < uint32_t(coarse_tri_array.size()); ++i) {
                coarse_all[i] = i;
            }
            std::vector<float3> coarse_positions;
            extract_positions(coarse.verts, coarse_positions);
            const auto parts = partitionTriangles(
                coarse_tri_array, coarse_all, coarse_positions,
                params.maxClusterTris);

            /* Cumulative error: a parent cluster must have an error at least as
             * large as all its children plus the current simplification error. */
            float max_child_error = 0.0f;
            for (const uint32_t cid : mg) {
                max_child_error = std::max(max_child_error, R.clusters[cid].error);
            }
            const float accumulated_error = max_child_error + std::max(simplified.error, 1e-4f);

            std::vector<uint32_t> parents;
            for (const auto &part : parts) {
                if (part.empty()) {
                    continue;
                }
                const uint32_t pid = makeCluster(
                    R, coarse, part, accumulated_error,
                    std::numeric_limits<float>::max(), level,
                    /*copy_verts=*/true);
                R.clusters[pid].material_index = group_mat;
                for (const uint32_t t : R.clusters[pid].tris) {
                    R.tris[t].material_index = group_mat;
                }
                parents.push_back(pid);
                next_frontier.push_back(pid);
            }
            if (parents.empty()) {
                continue;
            }

            /* Assign each child cluster to its nearest parent cluster and
             * link the DAG. The child's parent error is the error of the
             * simplification that produced the parent: it projects below the
             * pixel threshold exactly when the coarse version suffices. */
            for (const uint32_t cid : mg) {
                uint32_t best = parents.front();
                float best_d = std::numeric_limits<float>::max();
                for (const uint32_t pid : parents) {
                    const float d = math::distance_squared(
                        R.clusters[cid].sphereCenter,
                        R.clusters[pid].sphereCenter);
                    if (d < best_d) {
                        best_d = d;
                        best = pid;
                    }
                }
                R.clusters[cid].parentError = accumulated_error;
                R.clusters[best].children.push_back(cid);
            }
        }

        if (next_frontier.empty() || next_frontier.size() >= frontier.size()) {
            break;  /* no progress possible */
        }
        frontier = std::move(next_frontier);
        R.levelCount = level + 1;
    }
    return R;
}

}  // namespace detail
}  // namespace tessera
}  // namespace blender
