/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  LOD selection implementation: hierarchical traversal of the cluster DAG.
 *
 *  A cluster is emitted when the error of its *coarser parent* projects
 *  below the pixel-error threshold (exactly like UE5's rule): the coarse
 *  version is then good enough on screen, and drawing this (finer) cluster
 *  is visually sufficient without fetching the parent. Clusters whose parent
 *  is still too coarse are refined by descending into their children.
 *  Leaves are always emitted. Clusters outside the view frustum are culled
 *  together with their whole subtree.
 */

#include "nanite_lod.hh"

#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"

#include <cstddef>
#include <limits>
#include <vector>

namespace blender {
namespace nanite {

namespace {

/* Extract the six frustum planes (a*x + b*y + c*z + d >= 0 inside) from a
 * view-projection matrix. */
void extract_frustum_planes(const float4x4 &viewProj, float planes[6][4])
{
    /* Standard Gribb-Hartmann extraction. Blender matrices are column-major:
     * viewProj[col][row]; the i-th frustum plane is row i of the matrix
     * (sum or difference with row 3). */
    const float4x4 &m = viewProj;
    const auto row = [&](int r) {
        return float4(m[0][r], m[1][r], m[2][r], m[3][r]);
    };
    const float4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    planes[0][0] = r3.x + r0.x; planes[0][1] = r3.y + r0.y;
    planes[0][2] = r3.z + r0.z; planes[0][3] = r3.w + r0.w;  // left
    planes[1][0] = r3.x - r0.x; planes[1][1] = r3.y - r0.y;
    planes[1][2] = r3.z - r0.z; planes[1][3] = r3.w - r0.w;  // right
    planes[2][0] = r3.x + r1.x; planes[2][1] = r3.y + r1.y;
    planes[2][2] = r3.z + r1.z; planes[2][3] = r3.w + r1.w;  // bottom
    planes[3][0] = r3.x - r1.x; planes[3][1] = r3.y - r1.y;
    planes[3][2] = r3.z - r1.z; planes[3][3] = r3.w - r1.w;  // top
    planes[4][0] = r3.x + r2.x; planes[4][1] = r3.y + r2.y;
    planes[4][2] = r3.z + r2.z; planes[4][3] = r3.w + r2.w;  // near
    planes[5][0] = r3.x - r2.x; planes[5][1] = r3.y - r2.y;
    planes[5][2] = r3.z - r2.z; planes[5][3] = r3.w - r2.w;  // far
    /* Normalize the plane normals. */
    for (int p = 0; p < 6; ++p) {
        const float len = math::length(
            float3(planes[p][0], planes[p][1], planes[p][2]));
        if (len > 0.0f) {
            for (int k = 0; k < 4; ++k) {
                planes[p][k] /= len;
            }
        }
    }
}

// --- PART2 ---

/** Context shared by the traversal recursion. */
struct TraversalContext {
    const std::vector<NaniteCluster> *clusters = nullptr;
    const std::vector<std::vector<uint32_t>> *children = nullptr;
    float planes[6][4] = {};
    float3 camPos = float3(0.0f);
    float viewportHeight = 1.0f;
    float projScaleY = 1.0f;
    float pixelError = 1.0f;
    bool orthographic = false;
    std::size_t maxClusters = std::numeric_limits<std::size_t>::max();
    std::vector<NaniteDrawCommand> *out = nullptr;
    mutable bool budget_exhausted = false;
};

void traverseCluster(const TraversalContext &ctx, uint32_t ci)
{
    const NaniteCluster &cluster = (*ctx.clusters)[ci];

    /* Whole-subtree frustum culling. */
    const float3 center(cluster.sphere[0], cluster.sphere[1],
                        cluster.sphere[2]);
    if (!sphere_in_frustum(ctx.planes, center, cluster.sphere[3])) {
        return;
    }

    const auto &children = (*ctx.children)[ci];
    const bool has_children = !children.empty();

    /* Screen error of the coarser parent (FLT_MAX for roots -> always
     * refine). */
    const float parent_screen_err =
        (cluster.parentError >= std::numeric_limits<float>::max())
            ? std::numeric_limits<float>::max()
            : projectedErrorPx(center, cluster.sphere[3], cluster.parentError,
                               ctx.camPos, ctx.viewportHeight,
                               ctx.projScaleY, ctx.orthographic);

    /* Emit when the coarse version suffices, when the cluster itself is
     * already accurate enough, or when there is nothing finer to use. */
    const bool emit = !has_children || parent_screen_err <= ctx.pixelError;
    if (emit) {
        if (ctx.out->size() >= ctx.maxClusters) {
            ctx.budget_exhausted = true;
            return;
        }
        NaniteDrawCommand cmd = {};
        cmd.indexCount = cluster.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = cluster.firstIndex;
        cmd.vertexOffset = 0;
        cmd.firstInstance = 0;
        ctx.out->push_back(cmd);
        return;
    }

    for (const uint32_t child : children) {
        traverseCluster(ctx, child);
        if (ctx.budget_exhausted) {
            return;
        }
    }
}

}  // namespace

bool sphere_outside_plane(const float planes[4], const float3 &center,
                          float radius)
{
    const float dist = planes[0] * center.x + planes[1] * center.y +
                       planes[2] * center.z + planes[3];
    return dist < -radius;
}

bool sphere_in_frustum(const float planes[6][4], const float3 &center,
                       float radius)
{
    for (int p = 0; p < 6; ++p) {
        if (sphere_outside_plane(planes[p], center, radius)) {
            return false;
        }
    }
    return true;
}

float projectedErrorPx(const float3 &center, float radius, float error,
                       const float3 &camPos, float viewportHeight,
                       float projScaleY, bool orthographic)
{
    if (error <= 0.0f) {
        return 0.0f;
    }
    /* Perspective: divide by the distance to the cluster (minus the radius,
     * so distant-but-large clusters are not over-refined — same as UE).
     * Orthographic: pure scale, distance-independent. */
    const float dist = orthographic ?
                           1.0f :
                           std::max(math::distance(camPos, center) - radius,
                                    1e-6f);
    return error * projScaleY * (viewportHeight * 0.5f) / dist;
}

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

/* Invert a 4x4 matrix with partial pivoting (column-major, m[col][row]). */
static bool invert4x4(const float4x4 &m, float4x4 &out)
{
    double a[4][8];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            a[r][c] = double(m[c][r]);
            a[r][4 + c] = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) {
                piv = r;
            }
        }
        if (std::abs(a[piv][col]) < 1e-12) {
            return false;
        }
        if (piv != col) {
            for (int k = 0; k < 8; ++k) {
                std::swap(a[col][k], a[piv][k]);
            }
        }
        const double d = a[col][col];
        for (int k = 0; k < 8; ++k) {
            a[col][k] /= d;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) {
                continue;
            }
            const double f = a[r][col];
            for (int k = 0; k < 8; ++k) {
                a[r][k] -= f * a[col][k];
            }
        }
    }
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c][r] = float(a[r][4 + c]);
        }
    }
    return true;
}

/* Solve a 3x3 system with Cramer's rule (used to find the frustum apex). */
static bool solve3x3(const double A[3][3], const double b[3], double out[3])
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
        M[0][i] = b[0];
        M[1][i] = b[1];
        M[2][i] = b[2];
        const double sub = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        out[i] = sub / det;
    }
    return true;
}

NaniteViewInfo getViewInfo(const float4x4 &viewProj, float viewportHeight, float projScaleY)
{
    (void)viewportHeight;

    NaniteViewInfo view;
    extract_frustum_planes(viewProj, view.planes);

    /* Camera position: solve for the intersection of the frustum's side planes
     * (left, right, top). For perspective views this gives the exact camera
     * apex in the coordinate space of viewProj (e.g., object space if viewProj
     * includes the object matrix). For orthographic views the planes are parallel
     * so solve3x3 returns false and the error metric uses distance-independent scale. */
    const double A[3][3] = {
        {double(view.planes[0][0]), double(view.planes[0][1]), double(view.planes[0][2])},
        {double(view.planes[1][0]), double(view.planes[1][1]), double(view.planes[1][2])},
        {double(view.planes[3][0]), double(view.planes[3][1]), double(view.planes[3][2])}
    };
    const double b[3] = {-double(view.planes[0][3]), -double(view.planes[1][3]), -double(view.planes[3][3])};
    double apex[3];
    if (solve3x3(A, b, apex)) {
        view.camPos = float3(float(apex[0]), float(apex[1]), float(apex[2]));
        view.orthographic = false;
    }
    else {
        view.camPos = float3(0.0f);
        view.orthographic = true;
    }

    /* Vertical projection scale: taken from the projection matrix alone
     * (passed in as projScaleY). Deriving it from the combined viewProj is
     * incorrect when the object transform is baked into viewProj, because
     * object rotation corrupts viewProj[1][1]. */
    view.projScaleY = projScaleY;
    return view;
}

std::size_t selectClusters(const NaniteData &data,
                           const float4x4 &viewProj,
                           float projScaleY,
                           float viewportHeight,
                           const NaniteLodParams &params,
                           std::vector<NaniteDrawCommand> &out)
{
    out.clear();
    if (data.clusters.empty() || viewportHeight <= 0.0f) {
        return 0;
    }

    const NaniteViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);

    /* Build the DAG child lists from the parent links. */
    std::vector<std::vector<uint32_t>> children(data.clusters.size());
    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const int32_t pg = data.clusters[ci].parentGroup;
        if (pg >= 0 && uint32_t(pg) < data.clusters.size()) {
            children[uint32_t(pg)].push_back(ci);
        }
    }

    TraversalContext ctx;
    ctx.clusters = &data.clusters;
    ctx.children = &children;
    std::copy(&view.planes[0][0], &view.planes[0][0] + 24, &ctx.planes[0][0]);
    ctx.camPos = view.camPos;
    ctx.orthographic = view.orthographic;
    ctx.viewportHeight = viewportHeight;
    ctx.projScaleY = view.projScaleY;
    ctx.pixelError = params.pixelError;
    ctx.maxClusters = params.maxClustersPerFrame > 0
                          ? std::size_t(params.maxClustersPerFrame)
                          : std::numeric_limits<std::size_t>::max();
    ctx.out = &out;

    /* Traverse from every root (clusters without a parent). */
    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        if (data.clusters[ci].parentGroup == -1) {
            traverseCluster(ctx, ci);
            if (ctx.budget_exhausted) {
                break;
            }
        }
    }
    return out.size();
}

std::size_t selectClustersFlat(const NaniteData &data,
                               const float4x4 &viewProj,
                               float projScaleY,
                               float viewportHeight,
                               const NaniteLodParams &params,
                               std::vector<NaniteDrawCommand> &out)
{
    /* Fully parallelizable variant of the traversal rule: a cluster is drawn
     * when it has no children, or when the error of its coarser parent
     * projects below the pixel threshold. This is exactly the recursion
     * above, flattened — every cluster can be evaluated independently, which
     * is what the GPU compute traversal does (one thread per cluster). */
    out.clear();
    if (data.clusters.empty() || viewportHeight <= 0.0f) {
        return 0;
    }

    const NaniteViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);
    const std::size_t maxClusters = params.maxClustersPerFrame > 0
                                        ? std::size_t(params.maxClustersPerFrame)
                                        : std::numeric_limits<std::size_t>::max();

    /* Children counts and parent lookups from the parent links. */
    std::vector<uint32_t> children_count(data.clusters.size(), 0);
    for (const auto &c : data.clusters) {
        if (c.parentGroup >= 0 &&
            uint32_t(c.parentGroup) < data.clusters.size())
        {
            children_count[uint32_t(c.parentGroup)]++;
        }
    }

    /* Per-cluster emit predicate (flat mirror of the recursive traversal):
     *
     *   emit(C) = no ancestor of C passes its own test
     *             AND (leaf(C) OR test(C))
     *
     * where test(X) checks whether the error of X's parent projects below
     * the pixel threshold. The recursive traversal stops descending as soon
     * as an ancestor emits, so a cluster is only drawn when it is *reached*
     * (no ancestor emitted) and it is a leaf, or its own parent test passes.
     * The ancestor walk is bounded by the DAG depth (a handful of levels),
     * keeping the rule embarrassingly parallel — one thread per cluster on
     * the GPU, no queues. */
    const auto cluster_test = [&](const NaniteCluster &cluster) {
        if (cluster.parentError >= std::numeric_limits<float>::max()) {
            return false;
        }
        const float3 c(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        return projectedErrorPx(c, cluster.sphere[3], cluster.parentError,
                                view.camPos, viewportHeight, view.projScaleY,
                                view.orthographic) <= params.pixelError;
    };

    std::size_t emitted = 0;
    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const NaniteCluster &cluster = data.clusters[ci];
        /* Streaming residency: stub clusters (no vertex/index data) are
         * never drawn; their pages are requested by the page manager. */
        if (cluster.indexCount == 0) {
            continue;
        }
        const float3 center(cluster.sphere[0], cluster.sphere[1],
                            cluster.sphere[2]);
        const float radius = cluster.sphere[3];
        if (!sphere_in_frustum(view.planes, center, radius)) {
            continue;
        }

        /* Walk up the ancestor chain: if any ancestor would emit, the
         * recursive traversal stopped there and never reached this cluster;
         * if any ancestor's sphere is outside the frustum, the subtree was
         * pruned and this cluster is unreachable as well. */
        bool ancestor_emitted = false;
        int32_t ancestor = cluster.parentGroup;
        while (ancestor >= 0 &&
               uint32_t(ancestor) < data.clusters.size())
        {
            const NaniteCluster &a = data.clusters[uint32_t(ancestor)];
            if (a.indexCount == 0) {
                /* Non-resident ancestors cannot emit (UE-style: descend
                 * through missing pages). */
                ancestor = a.parentGroup;
                continue;
            }
            const float3 ac(a.sphere[0], a.sphere[1], a.sphere[2]);
            if (!sphere_in_frustum(view.planes, ac, a.sphere[3])) {
                ancestor_emitted = true;
                break;
            }
            if (cluster_test(a)) {
                ancestor_emitted = true;
                break;
            }
            ancestor = a.parentGroup;
        }
        if (ancestor_emitted) {
            continue;
        }

        const bool leaf = children_count[ci] == 0;
        const bool emit = leaf || cluster_test(cluster);
        if (!emit) {
            continue;
        }
        if (emitted >= maxClusters) {
            break;
        }
        NaniteDrawCommand cmd = {};
        cmd.indexCount = cluster.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = cluster.firstIndex;
        cmd.vertexOffset = 0;
        cmd.firstInstance = 0;
        out.push_back(cmd);
        emitted++;
    }
    return out.size();
}

}  // namespace nanite
}  // namespace blender