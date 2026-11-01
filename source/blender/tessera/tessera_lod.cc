/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
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

#include "tessera_lod.hh"

#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"

#include <cstddef>
#include <limits>
#include <vector>

namespace blender {
namespace tessera {

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
    const std::vector<TesseraCluster> *clusters = nullptr;
    const std::vector<std::vector<uint32_t>> *children = nullptr;
    float planes[6][4] = {};
    float3 camPos = float3(0.0f);
    float viewportHeight = 1.0f;
    float projScaleY = 1.0f;
    float pixelError = 1.0f;
    float cullingMargin = 0.0f;
    /* Reference instance scale applied to the source-unit cluster errors. */
    float error_scale = 1.0f;
    bool orthographic = false;
    std::size_t maxClusters = std::numeric_limits<std::size_t>::max();
    std::vector<TesseraDrawCommand> *out = nullptr;
    mutable bool budget_exhausted = false;
};

void traverseCluster(const TraversalContext &ctx, uint32_t ci)
{
    const TesseraCluster &cluster = (*ctx.clusters)[ci];

    /* Whole-subtree frustum culling. */
    const float3 center(cluster.sphere[0], cluster.sphere[1],
                        cluster.sphere[2]);
    if (!sphere_in_frustum(ctx.planes, center, cluster.sphere[3], ctx.cullingMargin)) {
        return;
    }

    const auto &children = (*ctx.children)[ci];
    const bool has_children = !children.empty();

    const float bound_dia_c = std::max(cluster.sphere[3] * 2.0f * ctx.error_scale, 0.001f);
    const float bound_dia_p = std::max(cluster.parentSphere[3] * 2.0f * ctx.error_scale, 0.001f);
    const float clean_err = std::min(cluster.error * ctx.error_scale, bound_dia_c);
    const float cluster_screen_err = (clean_err <= 0.0f)
        ? 0.0f
        : projectedErrorPx(center, cluster.sphere[3], clean_err,
                           ctx.camPos, ctx.viewportHeight,
                           ctx.projScaleY, ctx.orthographic);

    const float3 pcenter(cluster.parentSphere[0], cluster.parentSphere[1], cluster.parentSphere[2]);
    const float clean_perr = (cluster.parentError >= 1e38f)
        ? std::numeric_limits<float>::max()
        : std::min(cluster.parentError * ctx.error_scale, bound_dia_p);
    const float parent_screen_err =
        (cluster.parentError >= 1e38f)
            ? std::numeric_limits<float>::max()
            : projectedErrorPx(pcenter, cluster.parentSphere[3],
                               clean_perr,
                               ctx.camPos, ctx.viewportHeight,
                               ctx.projScaleY, ctx.orthographic);

    /* UE5 Tessera Cut Criterion:
     * Emit cluster if its error is acceptable (<= tau) AND its parent error is unacceptable (> tau). */
    if (cluster_screen_err <= ctx.pixelError && parent_screen_err > ctx.pixelError) {
        if (ctx.out->size() >= ctx.maxClusters) {
            ctx.budget_exhausted = true;
            return;
        }
        TesseraDrawCommand cmd = {};
        cmd.indexCount = cluster.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = cluster.firstIndex;
        cmd.vertexOffset = 0;
        cmd.firstInstance = 0;
        ctx.out->push_back(cmd);
        return;
    }

    if (cluster_screen_err > ctx.pixelError && has_children) {
        for (const uint32_t child : children) {
            traverseCluster(ctx, child);
            if (ctx.budget_exhausted) {
                return;
            }
        }
    }
}

}  // namespace

bool sphere_outside_plane(const float planes[4], const float3 &center,
                          float radius, float margin = 0.0f)
{
    const float dist = planes[0] * center.x + planes[1] * center.y +
                       planes[2] * center.z + planes[3];
    return dist < -(radius + margin);
}

bool sphere_in_frustum(const float planes[6][4], const float3 &center,
                       float radius, float margin)
{
    for (int p = 0; p < 6; ++p) {
        if (sphere_outside_plane(planes[p], center, radius, margin)) {
            return false;
        }
    }
    return true;
}

FrustumIntersection box_in_frustum(const float planes[6][4],
                                   const float3 &bmin,
                                   const float3 &bmax,
                                   float margin)
{
    const float3 center = (bmin + bmax) * 0.5f;
    const float3 extent = (bmax - bmin) * 0.5f;
    bool all_inside = true;
    for (int p = 0; p < 6; ++p) {
        const float dist = planes[p][0] * center.x + planes[p][1] * center.y +
                           planes[p][2] * center.z + planes[p][3];
        const float radius = std::abs(planes[p][0]) * extent.x +
                             std::abs(planes[p][1]) * extent.y +
                             std::abs(planes[p][2]) * extent.z;
        if (dist < -(radius + margin)) {
            return FrustumIntersection::OUTSIDE;
        }
        if (dist < radius - margin) {
            all_inside = false;
        }
    }
    return all_inside ? FrustumIntersection::INSIDE : FrustumIntersection::INTERSECTS;
}

void InstanceBVH::build(const std::vector<float4x4> &instance_mats,
                        const float3 &mesh_center,
                        float mesh_radius)
{
    clear();
    const size_t N = instance_mats.size();
    if (N == 0) {
        return;
    }
    built_instance_count = N;
    instance_indices.resize(N);
    for (size_t i = 0; i < N; ++i) {
        instance_indices[i] = uint32_t(i);
    }

    struct TempInst {
        float3 center;
        float radius;
    };
    std::vector<TempInst> inst_data(N);
    for (size_t i = 0; i < N; ++i) {
        const float4x4 &mat = instance_mats[i];
        inst_data[i].center = (mat * float4(mesh_center, 1.0f)).xyz();
        const float scale = std::max(math::length(mat[0].xyz()),
                                     std::max(math::length(mat[1].xyz()),
                                              math::length(mat[2].xyz())));
        inst_data[i].radius = mesh_radius * scale;
    }

    nodes.reserve(N * 2);

    struct BuildTask {
        int32_t node_idx;
        uint32_t start;
        uint32_t count;
    };

    InstanceBVHNode root;
    root.instance_start = 0;
    root.instance_count = uint32_t(N);
    nodes.push_back(root);

    std::vector<BuildTask> stack;
    stack.push_back({0, 0, uint32_t(N)});

    while (!stack.empty()) {
        BuildTask task = stack.back();
        stack.pop_back();

        float3 bmin(std::numeric_limits<float>::max());
        float3 bmax(std::numeric_limits<float>::lowest());
        float max_inst_r = 0.0f;
        for (uint32_t i = task.start; i < task.start + task.count; ++i) {
            const TempInst &inst = inst_data[instance_indices[i]];
            bmin.x = std::min(bmin.x, inst.center.x - inst.radius);
            bmin.y = std::min(bmin.y, inst.center.y - inst.radius);
            bmin.z = std::min(bmin.z, inst.center.z - inst.radius);
            bmax.x = std::max(bmax.x, inst.center.x + inst.radius);
            bmax.y = std::max(bmax.y, inst.center.y + inst.radius);
            bmax.z = std::max(bmax.z, inst.center.z + inst.radius);
            max_inst_r = std::max(max_inst_r, inst.radius);
        }
        nodes[task.node_idx].bmin = bmin;
        nodes[task.node_idx].bmax = bmax;
        nodes[task.node_idx].max_instance_radius = max_inst_r;
        nodes[task.node_idx].instance_start = task.start;
        nodes[task.node_idx].instance_count = task.count;

        if (task.count <= 16) {
            nodes[task.node_idx].left_child = -1;
            nodes[task.node_idx].right_child = -1;
            continue;
        }

        const float3 extent = bmax - bmin;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z) {
            axis = 1;
        }
        else if (extent.z > extent.x && extent.z > extent.y) {
            axis = 2;
        }

        const uint32_t mid = task.start + task.count / 2;
        std::nth_element(
            instance_indices.begin() + task.start,
            instance_indices.begin() + mid,
            instance_indices.begin() + task.start + task.count,
            [&](uint32_t a, uint32_t b) {
                if (axis == 0) return inst_data[a].center.x < inst_data[b].center.x;
                if (axis == 1) return inst_data[a].center.y < inst_data[b].center.y;
                return inst_data[a].center.z < inst_data[b].center.z;
            });

        int32_t left_idx = int32_t(nodes.size());
        nodes.emplace_back();
        int32_t right_idx = int32_t(nodes.size());
        nodes.emplace_back();

        nodes[task.node_idx].left_child = left_idx;
        nodes[task.node_idx].right_child = right_idx;

        stack.push_back({left_idx, task.start, mid - task.start});
        stack.push_back({right_idx, mid, task.start + task.count - mid});
    }
}

static inline float aabb_min_dist_sq(const float3 &p, const float3 &bmin, const float3 &bmax)
{
    float d2 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        if (p[i] < bmin[i]) {
            float diff = bmin[i] - p[i];
            d2 += diff * diff;
        } else if (p[i] > bmax[i]) {
            float diff = p[i] - bmax[i];
            d2 += diff * diff;
        }
    }
    return d2;
}

static inline float aabb_max_dist_sq(const float3 &p, const float3 &bmin, const float3 &bmax)
{
    float d2 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float d_min = std::abs(p[i] - bmin[i]);
        float d_max = std::abs(p[i] - bmax[i]);
        float diff = std::max(d_min, d_max);
        d2 += diff * diff;
    }
    return d2;
}

void InstanceBVH::cull(const TesseraViewInfo &view_info,
                       const std::vector<float4x4> &instance_mats,
                       const float3 &mesh_center,
                       float mesh_radius,
                       float culling_margin,
                       std::vector<float4x4> &out_visible,
                       float max_draw_dist,
                       float min_screen_radius_px,
                       float px_scale) const
{
    out_visible.clear();
    const size_t N = instance_mats.size();
    if (N == 0) {
        return;
    }
    const float max_dist_sq = (max_draw_dist > 0.0f) ? (max_draw_dist * max_draw_dist) : 0.0f;
    const float3 &cam_pos = view_info.camPos;

    if (!is_built() || N <= 16 || nodes.empty()) {
        /* Fallback linear culling for small instance counts */
        out_visible.reserve(N);
        for (const float4x4 &mat : instance_mats) {
            const float3 c = (mat * float4(mesh_center, 1.0f)).xyz();
            const float d_sq = math::distance_squared(c, cam_pos);
            if (max_dist_sq > 0.0f && d_sq > max_dist_sq) {
                continue;
            }
            if (min_screen_radius_px > 0.0f && px_scale > 0.0f && mesh_radius > 0.0f) {
                const float d = std::sqrt(std::max(0.0f, d_sq));
                const float inst_scale = std::max(
                    math::length(mat[0].xyz()),
                    std::max(math::length(mat[1].xyz()), math::length(mat[2].xyz())));
                const float world_radius = mesh_radius * inst_scale;
                const float screen_radius_px = (world_radius * px_scale) / std::max(d, 0.01f);
                if (screen_radius_px < min_screen_radius_px) {
                    continue;
                }
            }
            if (instance_in_frustum(view_info, mat, mesh_center, mesh_radius, culling_margin)) {
                out_visible.push_back(mat);
            }
        }
        return;
    }

    uint32_t stack[128];
    int stack_top = 0;
    stack[stack_top++] = 0;

    while (stack_top > 0) {
        const uint32_t node_idx = stack[--stack_top];
        const InstanceBVHNode &node = nodes[node_idx];

        const float min_d_sq = aabb_min_dist_sq(cam_pos, node.bmin, node.bmax);
        if (max_dist_sq > 0.0f && min_d_sq > max_dist_sq) {
            continue;
        }

        /* Sub-pixel subtree culling: if the largest instance in this subtree projects to less than
         * min_screen_radius_px at the closest point of the node AABB, discard ALL descendant
         * instances in O(1). */
        if (min_screen_radius_px > 0.0f && px_scale > 0.0f && node.max_instance_radius > 0.0f) {
            const float r_scaled = node.max_instance_radius * px_scale;
            if ((r_scaled * r_scaled) < (min_screen_radius_px * min_screen_radius_px) * std::max(min_d_sq, 0.0001f)) {
                continue;
            }
        }

        const FrustumIntersection isect = box_in_frustum(
            view_info.planes, node.bmin, node.bmax, culling_margin);

        if (isect == FrustumIntersection::OUTSIDE) {
            continue;
        }

        const bool all_within_dist = (max_dist_sq <= 0.0f) ||
                                     (aabb_max_dist_sq(cam_pos, node.bmin, node.bmax) <= max_dist_sq);

        if (isect == FrustumIntersection::INSIDE && all_within_dist && min_screen_radius_px <= 0.0f) {
            /* Bulk acceptance: all descendant instances are inside with zero further tests! */
            const uint32_t end = node.instance_start + node.instance_count;
            for (uint32_t i = node.instance_start; i < end; ++i) {
                out_visible.push_back(instance_mats[instance_indices[i]]);
            }
            continue;
        }

        /* Intersects: either test leaves or recurse */
        if (node.is_leaf()) {
            const uint32_t end = node.instance_start + node.instance_count;
            for (uint32_t i = node.instance_start; i < end; ++i) {
                const uint32_t inst_idx = instance_indices[i];
                const float4x4 &mat = instance_mats[inst_idx];
                const float3 c = (mat * float4(mesh_center, 1.0f)).xyz();
                const float d_sq = math::distance_squared(c, cam_pos);
                if (max_dist_sq > 0.0f && d_sq > max_dist_sq) {
                    continue;
                }
                if (min_screen_radius_px > 0.0f && px_scale > 0.0f && mesh_radius > 0.0f) {
                    const float inst_scale_sq = std::max(
                        math::length_squared(mat[0].xyz()),
                        std::max(math::length_squared(mat[1].xyz()), math::length_squared(mat[2].xyz())));
                    const float r_sq = (mesh_radius * mesh_radius) * inst_scale_sq;
                    const float r_scaled_sq = r_sq * (px_scale * px_scale);
                    if (r_scaled_sq < (min_screen_radius_px * min_screen_radius_px) * std::max(d_sq, 0.0001f)) {
                        continue;
                    }
                }
                if (isect == FrustumIntersection::INSIDE ||
                    instance_in_frustum(view_info, mat, mesh_center, mesh_radius, culling_margin))
                {
                    out_visible.push_back(mat);
                }
            }
        }
        else {
            if (stack_top + 2 < 128) {
                if (node.right_child >= 0) stack[stack_top++] = uint32_t(node.right_child);
                if (node.left_child >= 0) stack[stack_top++] = uint32_t(node.left_child);
            }
        }
    }
}

bool mesh_bounds_sphere(const TesseraData &data, float3 &r_center, float &r_radius)
{
    if (data.clusters.empty()) {
        return false;
    }
    /* O(1) fast path using precomputed header bounding box. */
    const float3 hmin(data.header.boundsMin[0], data.header.boundsMin[1], data.header.boundsMin[2]);
    const float3 hmax(data.header.boundsMax[0], data.header.boundsMax[1], data.header.boundsMax[2]);
    if (hmax.x > hmin.x || hmax.y > hmin.y || hmax.z > hmin.z) {
        r_center = (hmin + hmax) * 0.5f;
        r_radius = math::length(hmax - hmin) * 0.5f;
        if (r_radius > 0.0f) {
            return true;
        }
    }
    float bmin[3] = {std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
    float bmax[3] = {std::numeric_limits<float>::lowest(),
                     std::numeric_limits<float>::lowest(),
                     std::numeric_limits<float>::lowest()};
    for (const TesseraCluster &cluster : data.clusters) {
        const float3 c(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        const float r = cluster.sphere[3];
        for (int a = 0; a < 3; ++a) {
            bmin[a] = std::min(bmin[a], c[a] - r);
            bmax[a] = std::max(bmax[a], c[a] + r);
        }
    }
    const float3 lo(bmin[0], bmin[1], bmin[2]);
    const float3 hi(bmax[0], bmax[1], bmax[2]);
    r_center = (lo + hi) * 0.5f;
    r_radius = math::length(hi - lo) * 0.5f;
    return r_radius > 0.0f;
}

bool instance_in_frustum(const TesseraViewInfo &view_info,
                         const float4x4 &instance_mat,
                         const float3 &mesh_center,
                         float mesh_radius,
                         float culling_margin)
{
    /* World-space center of the transformed bounding sphere. */
    const float3 inst_center = (instance_mat * float4(mesh_center, 1.0f)).xyz();
    /* Worst-case radius scale: the largest axis length of the instance matrix. */
    const float inst_scale = std::max(math::length(instance_mat[0].xyz()),
                                      std::max(math::length(instance_mat[1].xyz()),
                                               math::length(instance_mat[2].xyz())));
    const float inst_radius = mesh_radius * inst_scale;
    return sphere_in_frustum(
        view_info.planes, inst_center, inst_radius, culling_margin);
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
                                    0.01f);
    return error * projScaleY * (viewportHeight * 0.5f) / dist;
}

float clusterScreenError(const TesseraCluster &cluster, float viewportHeight)
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

TesseraViewInfo getViewInfo(const float4x4 &viewProj, float viewportHeight, float projScaleY)
{
    (void)viewportHeight;

    TesseraViewInfo view;
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

std::size_t selectClusters(const TesseraData &data,
                           const float4x4 &viewProj,
                           float projScaleY,
                           float viewportHeight,
                           const TesseraLodParams &params,
                           std::vector<TesseraDrawCommand> &out)
{
    out.clear();
    if (data.clusters.empty() || viewportHeight <= 0.0f) {
        return 0;
    }

    const TesseraViewInfo view = getViewInfo(viewProj, viewportHeight, projScaleY);

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
    ctx.cullingMargin = params.cullingMargin;
    ctx.error_scale = params.errorScale > 0.0f ? params.errorScale : 1.0f;
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

std::size_t selectClustersFlat(const TesseraData &data,
                               const float4x4 &viewProj,
                               float projScaleY,
                               float viewportHeight,
                               const TesseraLodParams &params,
                               std::vector<TesseraDrawCommand> &out,
                               const float4x4 *lodViewProj,
                               const float *lodProjScaleY)
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

    const TesseraViewInfo draw_view = getViewInfo(viewProj, viewportHeight, projScaleY);
    const TesseraViewInfo lod_view = (lodViewProj != nullptr) ?
        getViewInfo(*lodViewProj, viewportHeight, (lodProjScaleY != nullptr) ? *lodProjScaleY : projScaleY) :
        draw_view;
    const TesseraViewInfo &culling_view = (lodViewProj != nullptr) ? lod_view : draw_view;

    const std::size_t maxClusters = params.maxClustersPerFrame > 0
                                        ? std::size_t(params.maxClustersPerFrame)
                                        : std::numeric_limits<std::size_t>::max();

    std::size_t emitted = 0;
    for (uint32_t ci = 0; ci < uint32_t(data.clusters.size()); ++ci) {
        const TesseraCluster &cluster = data.clusters[ci];
        /* Streaming residency: stub clusters (no vertex/index data) are
         * never drawn; their pages are requested by the page manager. */
        if (cluster.indexCount == 0) {
            continue;
        }
        const float3 center(cluster.sphere[0], cluster.sphere[1], cluster.sphere[2]);
        const float radius = cluster.sphere[3];
        if (!sphere_in_frustum(culling_view.planes, center, radius, params.cullingMargin)) {
            continue;
        }

        /* Tessera DAG Cut Criterion:
         * A cluster is rendered iff:
         *   projectedError(cluster) <= tau  AND  projectedError(parent) > tau
         */
        const float scale = params.errorScale > 0.0f ? params.errorScale : 1.0f;
        const float pradius = cluster.parentSphere[3];
        const float bound_dia_c = std::max(radius * 2.0f * scale, 0.001f);
        const float bound_dia_p = std::max(pradius * 2.0f * scale, 0.001f);
        const float clean_err = std::min(cluster.error * scale, bound_dia_c);
        const float err_c = (clean_err <= 0.0f)
            ? 0.0f
            : projectedErrorPx(center, radius, clean_err,
                               lod_view.camPos, viewportHeight, lod_view.projScaleY,
                               lod_view.orthographic);

        const float3 pcenter(cluster.parentSphere[0], cluster.parentSphere[1], cluster.parentSphere[2]);
        const float clean_perr = (cluster.parentError >= 1e38f)
            ? std::numeric_limits<float>::max()
            : std::min(cluster.parentError * scale, bound_dia_p);
        const float err_p = (cluster.parentError >= 1e38f)
            ? std::numeric_limits<float>::max()
            : projectedErrorPx(pcenter, pradius, clean_perr,
                               lod_view.camPos, viewportHeight, lod_view.projScaleY,
                               lod_view.orthographic);

        const float dist_c = std::max(math::distance(lod_view.camPos, center) - radius, 0.01f);
        const float safe_lod_dist = std::max(params.lodDistance, 0.1f);
        const float d_norm = dist_c / safe_lod_dist;
        const float tau_effective = params.pixelError * (1.0f + std::min(d_norm * d_norm * d_norm, 500.0f) * params.lodIntensity);

        /* Mid/Far distance cluster level clamping: for clusters beyond 5/9/14 meters,
         * prevent selection of fine Level 0..2 clusters when a coarser parent exists. */
        if (dist_c > 5.0f && cluster.level == 0 && cluster.parentGroup != -1) {
            continue;
        }
        if (dist_c > 9.0f && cluster.level <= 1 && cluster.parentGroup != -1) {
            continue;
        }
        if (dist_c > 14.0f && cluster.level <= 2 && cluster.parentGroup != -1) {
            continue;
        }

        if (err_c <= tau_effective && err_p > tau_effective) {
            if (emitted >= maxClusters) {
                break;
            }
            TesseraDrawCommand cmd = {};
            cmd.indexCount = cluster.indexCount;
            cmd.instanceCount = 1;
            cmd.firstIndex = cluster.firstIndex;
            cmd.vertexOffset = 0;
            cmd.firstInstance = ci;
            out.push_back(cmd);
            emitted++;
        }
    }
    return out.size();
}

}  // namespace tessera
}  // namespace blender
