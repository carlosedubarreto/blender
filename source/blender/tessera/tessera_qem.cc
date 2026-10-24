/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Quadric error metric (QEM) edge-collapse simplification.
 */

#include "tessera_qem.hh"

#include "BLI_math_vector.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace blender {
namespace tessera {
namespace detail {

namespace {

/* Symmetric 4x4 quadric, stored as the upper triangle:
 * [ xx  xy  xz  xw ]
 * [     yy  yz  yw ]
 * [         zz  zw ]
 * [             ww ] */
struct Quadric {
    double q[10] = {0};

    void add_plane(double a, double b, double c, double d, double w)
    {
        q[0] += w * a * a;
        q[1] += w * a * b;
        q[2] += w * a * c;
        q[3] += w * a * d;
        q[4] += w * b * b;
        q[5] += w * b * c;
        q[6] += w * b * d;
        q[7] += w * c * c;
        q[8] += w * c * d;
        q[9] += w * d * d;
    }

    Quadric &operator+=(const Quadric &other)
    {
        for (int i = 0; i < 10; ++i) {
            q[i] += other.q[i];
        }
        return *this;
    }
};

uint64_t edge_key(uint32_t a, uint32_t b)
{
    if (a > b) {
        std::swap(a, b);
    }
    return (static_cast<uint64_t>(a) << 32) | b;
}

double quadric_cost(const Quadric &Q, const float3 &v)
{
    const double x = v.x, y = v.y, z = v.z;
    return Q.q[0] * x * x + Q.q[1] * x * y + Q.q[2] * x * z + Q.q[3] * x +
           Q.q[4] * y * y + Q.q[5] * y * z + Q.q[6] * y + Q.q[7] * z * z +
           Q.q[8] * z + Q.q[9];
}

/* Solve the 3x3 linear system A v = b with Cramer's rule. */
bool solve3(const double A[3][3], const double b[3], double out[3])
{
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                       A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                       A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (std::abs(det) < 1e-12) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        double M[3][3] = {A[0][0], A[0][1], A[0][2], A[1][0], A[1][1],
                          A[1][2], A[2][0], A[2][1], A[2][2]};
        M[i][0] = b[0];
        M[i][1] = b[1];
        M[i][2] = b[2];
        const double sub = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        out[i] = sub / det;
    }
    return true;
}

/* Optimal vertex position minimizing the quadric error. */
bool optimal_position(const Quadric &Q, float3 &out)
{
    const double A[3][3] = {{Q.q[0], Q.q[1], Q.q[2]},
                            {Q.q[1], Q.q[4], Q.q[5]},
                            {Q.q[2], Q.q[5], Q.q[7]}};
    const double b[3] = {-Q.q[3], -Q.q[6], -Q.q[8]};
    double p[3];
    if (!solve3(A, b, p)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(p[i])) {
            return false;
        }
    }
    out = float3(float(p[0]), float(p[1]), float(p[2]));
    return true;
}

struct QueueEntry {
    double cost;
    uint64_t key;
    double penalty = 0.0;
    bool operator>(const QueueEntry &o) const { return cost > o.cost; }
};

}  // namespace

// --- PART2 ---

SimplifyResult simplifyTriangles(const std::vector<float3> &positions,
                                 const std::vector<uint32_t> &tri_verts,
                                 uint32_t target_tri_count)
{
    SimplifyResult result;
    const uint32_t ntris = uint32_t(tri_verts.size() / 3);
    result.tris = tri_verts;
    if (ntris == 0 || tri_verts.size() % 3 != 0) {
        return result;
    }
    if (target_tri_count >= ntris) {
        return result;
    }

    /* Compact the referenced vertices into a local array so simplification
     * can freely move them without touching geometry owned by other
     * clusters. */
    std::unordered_map<uint32_t, uint32_t> to_local;
    to_local.reserve(tri_verts.size());
    std::vector<float3> verts;
    verts.reserve(tri_verts.size());
    std::vector<std::array<uint32_t, 3>> tris(ntris);
    for (uint32_t t = 0; t < ntris; ++t) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t g = tri_verts[t * 3 + k];
            auto it = to_local.find(g);
            if (it == to_local.end()) {
                const uint32_t local = uint32_t(verts.size());
                to_local.emplace(g, local);
                verts.push_back(positions[g]);
            }
            tris[t][k] = to_local[g];
        }
    }

    const uint32_t nv = uint32_t(verts.size());

    /* Edge usage counts within this set; edges used by a single triangle are
     * the open boundary of the patch. Boundary vertices are locked so the
     * patch border stays compatible with neighboring clusters. */
    std::unordered_map<uint64_t, uint32_t> edge_count;
    edge_count.reserve(ntris * 3);
    for (const auto &t : tris) {
        for (int k = 0; k < 3; ++k) {
            edge_count[edge_key(t[k], t[(k + 1) % 3])]++;
        }
    }
    std::vector<char> locked(nv, 0);
    for (const auto &e : edge_count) {
        if (e.second == 1) {
            locked[e.first >> 32] = 1;
            locked[e.first & 0xffffffff] = 1;
        }
    }

    // --- PART3 ---

    /* Vertex -> adjacent triangles, face normals, per-vertex quadrics. */
    std::vector<std::vector<uint32_t>> vtris(nv);
    std::vector<float3> fnorm(ntris);
    std::vector<Quadric> Q(nv);
    for (uint32_t t = 0; t < ntris; ++t) {
        const auto &tv = tris[t];
        if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
            continue;
        }
        const float3 &p0 = verts[tv[0]];
        const float3 &p1 = verts[tv[1]];
        const float3 &p2 = verts[tv[2]];
        const float3 un = math::cross(p1 - p0, p2 - p0);
        if (math::length_squared(un) <= 1e-20f) {
            /* Zero-area (coincident or collinear positions): dead. */
            tris[t] = {tv[0], tv[0], tv[0]};
            continue;
        }
        fnorm[t] = math::normalize(un);
        const double w = double(math::length(un)) * 0.5 + 1e-12;
        const double d = -double(math::dot(fnorm[t], p0));
        for (int k = 0; k < 3; ++k) {
            vtris[tv[k]].push_back(t);
        }
        Q[tv[0]].add_plane(fnorm[t].x, fnorm[t].y, fnorm[t].z, d, w);
        Q[tv[1]].add_plane(fnorm[t].x, fnorm[t].y, fnorm[t].z, d, w);
        Q[tv[2]].add_plane(fnorm[t].x, fnorm[t].y, fnorm[t].z, d, w);
    }

    /* Live edge set. */
    std::unordered_set<uint64_t> live;
    live.reserve(edge_count.size());
    for (const auto &e : edge_count) {
        live.insert(e.first);
    }

    auto edge_cost = [&](uint32_t a, uint32_t b) {
        Quadric Qe = Q[a];
        Qe += Q[b];
        float3 pos;
        if (!optimal_position(Qe, pos)) {
            pos = (verts[a] + verts[b]) * 0.5f;
        }
        const double cost = std::max(quadric_cost(Qe, pos), 0.0);
        return std::isfinite(cost) ? cost : 1e30;
    };

    std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                        std::greater<QueueEntry>>
        queue;
    auto push_edge = [&](uint32_t a, uint32_t b) {
        const uint64_t key = edge_key(a, b);
        if (live.count(key)) {
            queue.push({edge_cost(a, b), key, 0.0});
        }
    };
    for (const uint64_t key : live) {
        queue.push({edge_cost(key >> 32, key & 0xffffffff), key, 0.0});
    }

    uint32_t live_tris = ntris;
    double max_err = 0.0;

    // --- PART4 ---

    live_tris = 0;
    for (uint32_t t = 0; t < ntris; ++t) {
        const auto &tv = tris[t];
        if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
            continue;
        }
        live_tris++;
    }

    /* Rebuild the quadric of a vertex from its adjacent live triangles. */
    auto rebuild_quadric = [&](uint32_t v) {
        Q[v] = Quadric();
        for (const uint32_t t : vtris[v]) {
            const auto &tv = tris[t];
            if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                continue;
            }
            const float3 &p0 = verts[tv[0]];
            const float3 &p1 = verts[tv[1]];
            const float3 &p2 = verts[tv[2]];
            const float3 un = math::cross(p1 - p0, p2 - p0);
            if (math::length_squared(un) <= 0.0f) {
                continue;
            }
            const float3 n = math::normalize(un);
            const double w = double(math::length(un)) * 0.5 + 1e-12;
            const double d = -double(math::dot(n, p0));
            Q[v].add_plane(n.x, n.y, n.z, d, w);
        }
    };

    /* True when moving vertex `v` of triangle `t` to `newp` degenerates or
     * flips the triangle's normal. */
    auto flips = [&](uint32_t t, uint32_t v, const float3 &newp) {
        const auto &tv = tris[t];
        float3 p[3];
        for (int k = 0; k < 3; ++k) {
            p[k] = verts[tv[k]];
            if (tv[k] == v) {
                p[k] = newp;
            }
        }
        const float3 un = math::cross(p[1] - p[0], p[2] - p[0]);
        if (math::length_squared(un) <= 1e-20f) {
            return true;
        }
        return math::dot(math::normalize(un), fnorm[t]) <= 0.05f;
    };

    // --- PART5 ---

    /* Collapse loop. When the queue drains before reaching the target
     * (transient rejections), the queue is rebuilt from all live edges with
     * up-to-date costs — a few passes unlock edges that were only rejected
     * because neighbors had not collapsed yet. */
    int rebuild_passes = 3;
    while (live_tris > target_tri_count) {
        if (queue.empty()) {
            if (rebuild_passes-- <= 0) {
                break;
            }
            for (const uint64_t key : live) {
                queue.push({edge_cost(key >> 32, key & 0xffffffff), key, 0.0});
            }
            if (queue.empty()) {
                break;
            }
        }
        const QueueEntry entry = queue.top();
        queue.pop();
        const uint32_t a = uint32_t(entry.key >> 32);
        const uint32_t b = uint32_t(entry.key & 0xffffffff);
        if (!live.count(entry.key)) {
            continue;
        }
        const double cur = edge_cost(a, b);
        if (std::abs((cur + entry.penalty) - entry.cost) >
            1e-9 * (1.0 + std::abs(entry.cost)))
        {
            queue.push({cur, entry.key, entry.penalty}); /* stale, re-rank */
            continue;
        }
        /* Locked boundary endpoints never move (keeps cluster borders
         * watertight-compatible). */
        if (locked[a] && locked[b]) {
            live.erase(entry.key);
            continue;
        }

        Quadric Qab = Q[a];
        Qab += Q[b];
        float3 newp;
        if (locked[a]) {
            newp = verts[a];
        }
        else if (locked[b]) {
            newp = verts[b];
        }
        else if (!optimal_position(Qab, newp)) {
            newp = (verts[a] + verts[b]) * 0.5f;
        }
        /* The surviving vertex a inherits b's boundary lock: when the locked
         * endpoint b is removed, a takes its place on the border and must be
         * frozen there too (otherwise later collapses would deform the
         * cluster boundary and break watertightness). */
        if (locked[b]) {
            locked[a] = 1;
        }

        bool ok = true;
        for (const uint32_t t : vtris[a]) {
            const auto &tv = tris[t];
            if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                continue;
            }
            const bool has_b = (tv[0] == b || tv[1] == b || tv[2] == b);
            if (!has_b && flips(t, a, newp)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            for (const uint32_t t : vtris[b]) {
                const auto &tv = tris[t];
                if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                    continue;
                }
                const bool has_a = (tv[0] == a || tv[1] == a || tv[2] == a);
                if (!has_a && flips(t, b, newp)) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            /* Rejected (flip/topology): drop this entry; the pass-based
             * rebuild below retries it with updated geometry later. */
            continue;
        }

        // --- PART6 ---

        /* Perform the collapse: b merges into a at newp. */
        max_err = std::max(max_err, std::sqrt(std::max(cur, 0.0)));
        verts[a] = newp;
        for (const uint32_t t : vtris[b]) {
            auto &tv = tris[t];
            if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                continue;
            }
            bool removed = false;
            for (int k = 0; k < 3; ++k) {
                if (tv[k] == b) {
                    tv[k] = a;
                }
            }
            for (int k = 0; k < 3; ++k) {
                if (tv[k] == a && tv[(k + 1) % 3] == a) {
                    removed = true;
                    break;
                }
            }
            if (removed) {
                tris[t] = {a, a, a};
                live_tris--;
                continue;
            }
            vtris[a].push_back(t);
            const float3 &p0 = verts[tv[0]];
            const float3 &p1 = verts[tv[1]];
            const float3 &p2 = verts[tv[2]];
            const float3 un = math::cross(p1 - p0, p2 - p0);
            if (math::length_squared(un) <= 1e-20f) {
                /* Zero-area after the collapse (distinct ids at coincident
                 * or collinear positions): dead. */
                tris[t] = {a, a, a};
                live_tris--;
                continue;
            }
            fnorm[t] = math::normalize(un);
        }
        vtris[b].clear();
        live.erase(entry.key);

        /* Rebuild quadrics around the merged vertex and refresh the cost of
         * all affected edges. */
        std::unordered_set<uint32_t> touched;
        for (const uint32_t t : vtris[a]) {
            const auto &tv = tris[t];
            if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                continue;
            }
            touched.insert(tv[0]);
            touched.insert(tv[1]);
            touched.insert(tv[2]);
        }
        for (const uint32_t v : touched) {
            rebuild_quadric(v);
            for (const uint32_t t : vtris[v]) {
                const auto &tv = tris[t];
                if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
                    continue;
                }
                push_edge(tv[0], tv[1]);
                push_edge(tv[1], tv[2]);
                push_edge(tv[2], tv[0]);
            }
        }
    }

    /* Collect surviving triangles in local space (positions are returned in
     * #result.positions, triangles reference it directly). */
    result.positions = std::move(verts);
    result.tris.clear();
    result.tris.reserve(size_t(live_tris) * 3);
    for (uint32_t t = 0; t < ntris; ++t) {
        const auto &tv = tris[t];
        if (tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2]) {
            continue;
        }
        const float3 &p0 = result.positions[tv[0]];
        const float3 &p1 = result.positions[tv[1]];
        const float3 &p2 = result.positions[tv[2]];
        if (math::length_squared(math::cross(p1 - p0, p2 - p0)) <= 1e-20f) {
            continue;  /* zero-area debris */
        }
        result.tris.push_back(tv[0]);
        result.tris.push_back(tv[1]);
        result.tris.push_back(tv[2]);
    }
    result.error = float(max_err);
    return result;
}

}  // namespace detail
}  // namespace tessera
}  // namespace blender
