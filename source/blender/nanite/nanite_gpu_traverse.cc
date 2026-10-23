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

  uvec4 meta = cluster_meta[ci];
  uint index_count = meta.y;
  if (index_count == 0u) {
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

  /* Nanite DAG Cut Criterion:
   * A cluster is rendered iff: E_cluster <= tau AND E_parent > tau */
  vec2 errs = cluster_errors[ci];
  float err_c = errs.x;
  float err_p = errs.y;

  float dist_c = max(distance(camera_pos, center) - radius, 1e-4);
  float screen_err_c = (err_c <= 0.0) ? 0.0 : (err_c * proj_scale_y * (viewport_height * 0.5) / dist_c);

  vec4 parent_sphere = cluster_parent_sphere[ci];
  float dist_p = max(distance(camera_pos, parent_sphere.xyz) - parent_sphere.w, 1e-4);
  float screen_err_p = (err_p >= 3.0e38) ? 1.0e39 : (err_p * proj_scale_y * (viewport_height * 0.5) / dist_p);

  if (!(screen_err_c <= pixel_error && screen_err_p > pixel_error)) {
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

    /* NOTE: The info name must be "pyGPU_Shader": the python-path shader
     * compiler expands resource declarations through the hardcoded
     * CREATE_INFO_RES_PASS_pyGPU_Shader hook. */
    gshader::ShaderCreateInfo info("pyGPU_Shader");
    info.local_group_size(64, 1, 1);
    /* The SSBO element types below are plain GLSL scalar/vector types; flag
     * the info so they are not confused with host-shared struct linting. */
    info.builtins_ |= gshader::BuiltinBits::NO_BUFFER_TYPE_LINTING;

    /* binding 0: bounding spheres (xyz = center, w = radius). */
    info.storage_buf(0, gshader::Qualifier::read, "vec4", "cluster_sphere[]");
    /* binding 1: parent bounding spheres (xyz = center, w = radius). */
    info.storage_buf(1, gshader::Qualifier::read, "vec4", "cluster_parent_sphere[]");
    /* binding 2: cluster error (x) and parent error (y). */
    info.storage_buf(2, gshader::Qualifier::read, "vec2", "cluster_errors[]");
    /* binding 3: x = firstIndex, y = indexCount, z = childrenCount,
     * w = encoded parent index (0xFFFFFFFF for roots). */
    info.storage_buf(3, gshader::Qualifier::read, "uvec4", "cluster_meta[]");
    /* binding 4: indirect draw commands (4 uints per command). */
    info.storage_buf(4, gshader::Qualifier::write, "uint", "draw_commands[]");
    /* binding 5: shared atomic draw counter. */
    info.storage_buf(5, gshader::Qualifier::read_write, "uint", "draw_counter");

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
    return GPU_shader_create_from_info_python(raw);
}

}  // namespace nanite
}  // namespace blender
