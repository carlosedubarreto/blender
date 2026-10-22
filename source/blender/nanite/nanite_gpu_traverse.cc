/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Phase 1: GPU-driven traversal implementation.
 */

#include "nanite_gpu_traverse.hh"

#include "GPU_shader_create_info.hh"

namespace blender {
namespace nanite {

gpu::Shader *createTraverseShader()
{
    namespace gshader = blender::gpu::shader;

    static const char *COMPUTE_SOURCE = R"(
void main()
{
  uint ci = uint(gl_GlobalInvocationID.x);
  if (ci >= uint(cluster_count)) {
    return;
  }

  vec4 sphere = cluster_sphere[ci];
  vec3 center = sphere.xyz;
  float radius = sphere.w;

  /* Frustum culling (Gribb-Hartmann planes, a*x + b*y + c*z + d >= 0 inside). */
  for (int p = 0; p < 6; ++p) {
    float d = dot(frustum_planes[p].xyz, center) + frustum_planes[p].w;
    if (d < -radius) {
      return;
    }
  }

  /* Flat LOD rule, one thread per cluster (equivalent to the recursive
   * traversal):
   *
   *   emit(C) = no ancestor of C passes its own test
   *             AND (leaf(C) OR test(C))
   *
   * where test(X) projects the error of X's parent to pixels. The recursive
   * traversal stops descending as soon as an ancestor emits, so a cluster is
   * only drawn when it is *reached* (no ancestor emitted) and it is a leaf,
   * or its own parent test passes. The ancestor walk is bounded by the DAG
   * depth, keeping the rule embarrassingly parallel. */
  uvec4 meta = cluster_meta[ci];
  uint parent = meta.w;

  float err_c = cluster_parent_err[ci];
  bool test_c = false;
  if (err_c < 3.0e38) {
    float dist = max(distance(camera_pos, center) - radius, 1e-6);
    float screen_err = err_c * proj_scale_y * (viewport_height * 0.5) / dist;
    test_c = screen_err <= pixel_error;
  }

  /* Walk up the ancestor chain: if any ancestor would emit, the recursive
   * traversal stopped there and never reached this cluster; if any
   * ancestor's sphere is outside the frustum, the subtree was pruned.
   * Non-resident ancestors (streaming stubs, indexCount == 0) cannot emit,
   * so the walk descends through them. */
  uint ancestor = meta.w;
  bool ancestor_emitted = false;
  for (int depth = 0; depth < 64 && ancestor != 0xFFFFFFFFu; ++depth) {
    uvec4 ameta = cluster_meta[ancestor];
    if (ameta.y == 0u) {
      /* Non-resident stub. */
      ancestor = ameta.w;
      continue;
    }
    vec4 asphere = cluster_sphere[ancestor];
    vec3 acenter = asphere.xyz;
    bool a_outside = false;
    for (int p = 0; p < 6; ++p) {
      float d = dot(frustum_planes[p].xyz, acenter) + frustum_planes[p].w;
      if (d < -asphere.w) {
        a_outside = true;
        break;
      }
    }
    if (a_outside) {
      ancestor_emitted = true;
      break;
    }
    float err_a = cluster_parent_err[ancestor];
    if (err_a < 3.0e38) {
      float adist = max(distance(camera_pos, acenter) - asphere.w, 1e-6);
      float ascreen = err_a * proj_scale_y * (viewport_height * 0.5) / adist;
      if (ascreen <= pixel_error) {
        ancestor_emitted = true;
        break;
      }
    }
    ancestor = cluster_meta[ancestor].w;
  }
  if (ancestor_emitted) {
    return;
  }

  bool leaf = (meta.z == 0u);
  bool resident = (meta.y != 0u);
  if (!(resident && (leaf || test_c))) {
    return;
  }

  /* Budget + overflow guards (read the shared counter before allocating). */
  if ((max_clusters > 0) && (draw_counter >= uint(max_clusters))) {
    return;
  }
  uint slot = atomicAdd(draw_counter, 1u);
  if (slot >= uint(max_commands)) {
    return;
  }

  /* DrawArraysIndirectCommand layout (4 x uint32): the visibility pass uses
   * vertex pulling, so `first` is the absolute index-buffer position and the
   * triangle id is derived from gl_VertexID directly. */
  uint base = slot * 4u;
  draw_commands[base + 0u] = meta.y;
  draw_commands[base + 1u] = 1u;
  draw_commands[base + 2u] = meta.x;
  draw_commands[base + 3u] = 0u;
}
)";

    gshader::ShaderCreateInfo info("nanite_traverse");
    info.local_group_size(64, 1, 1);
    /* The SSBO element types below are plain GLSL scalar/vector types; flag
     * the info so they are not confused with host-shared struct linting. */
    info.builtins_ |= gshader::BuiltinBits::NO_BUFFER_TYPE_LINTING;

    /* binding 0: bounding spheres (xyz = center, w = radius). */
    info.storage_buf(0, gshader::Qualifier::read, "vec4", "cluster_sphere[]");
    /* binding 1: parent simplification errors. */
    info.storage_buf(1, gshader::Qualifier::read, "float", "cluster_parent_err[]");
    /* binding 2: x = firstIndex, y = indexCount, z = childrenCount,
     * w = encoded parent index (0xFFFFFFFF for roots). */
    info.storage_buf(2, gshader::Qualifier::read, "uvec4", "cluster_meta[]");
    /* binding 3: indirect draw commands (5 uints per command). */
    info.storage_buf(3, gshader::Qualifier::write, "uint", "draw_commands[]");
    /* binding 4: shared atomic draw counter. */
    info.storage_buf(4, gshader::Qualifier::read_write, "uint", "draw_counter");

    info.push_constant(gshader::Type::float3_t, "camera_pos");
    info.push_constant(gshader::Type::float_t, "viewport_height");
    info.push_constant(gshader::Type::float_t, "proj_scale_y");
    info.push_constant(gshader::Type::float_t, "pixel_error");
    info.push_constant(gshader::Type::int_t, "cluster_count");
    info.push_constant(gshader::Type::int_t, "max_clusters");
    info.push_constant(gshader::Type::int_t, "max_commands");
    info.push_constant(gshader::Type::float4_t, "frustum_planes", 6);

    /* Runtime-created shader: register the raw source as "generated" content
     * (same mechanism as the `gpu` Python module). */
    info.compute_source("nanite_traverse.glsl");
    info.compute_source_generated = COMPUTE_SOURCE;

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info(raw);
}

}  // namespace nanite
}  // namespace blender
