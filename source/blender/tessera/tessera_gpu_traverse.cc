/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 1: GPU-driven traversal implementation.
 */

#include "tessera_gpu_traverse.hh"

#include "GPU_shader_create_info.hh"

namespace blender {
namespace tessera {

gpu::Shader *createTraverseShader()
{
    namespace gshader = blender::gpu::shader;

    static const char *COMPUTE_SOURCE = R"(
/* Index-count buckets for the Solid-Mode single-indirect-draw path: each
 * visible cluster is appended to the bucket matching its index count so the
 * per-bucket DrawArraysIndirectCommand can use a vertexCount close to the
 * real cluster size (a single max-size draw would launch up to max/24 times
 * more vertex shaders than needed). Sizes are fixed; the last bucket holds
 * every cluster larger than 384 indices and is sized by the `bucket_max`
 * push constant (the mesh's true max cluster index count). */
uint bucket_of(uint index_count)
{
  if (index_count <= 24u) { return 0u; }
  if (index_count <= 48u) { return 1u; }
  if (index_count <= 96u) { return 2u; }
  if (index_count <= 192u) { return 3u; }
  if (index_count <= 384u) { return 4u; }
  return 5u;
}

uint bucket_size(uint b)
{
  if (b == 0u) { return 24u; }
  if (b == 1u) { return 48u; }
  if (b == 2u) { return 96u; }
  if (b == 3u) { return 192u; }
  if (b == 4u) { return 384u; }
  return uint(bucket_max);
}

void main()
{
  uint ci = uint(gl_GlobalInvocationID.x);
  if (ci == 0u && finalize_pass != 0) {
    /* Finalize pass (this shader re-dispatched with cluster_count == 0 after
     * the main traversal, so only thread 0 runs). Write one
     * DrawArraysIndirectCommand per bucket at draw_counters[84 + b*4 .. +3]:
     *   [84+b*4+0] = vertexCount   (bucket size)
     *   [84+b*4+1] = instanceCount (bucket visible count * object instances)
     *   [84+b*4+2] = firstVertex   (0)
     *   [84+b*4+3] = firstInstance (0)
     * The per-bucket visible counts were accumulated atomically at
     * draw_counters[76 + b] by the main dispatch; multiplying here (after
     * all traversal threads finished) scales the draw for scattered fields
     * where every object instance reuses the same cluster selection. */
    for (int b = 0; b < 6; ++b) {
      uint count = draw_counters[76 + b];
      draw_counters[84 + b * 4 + 0] = bucket_size(uint(b));
      draw_counters[84 + b * 4 + 1] = count * uint(finalize_multiplier);
      draw_counters[84 + b * 4 + 2] = 0u;
      draw_counters[84 + b * 4 + 3] = 0u;
    }
  }
  if (ci >= uint(cluster_count)) {
    return;
  }

  uint cluster_index = ci;
  if (max_valid_clusters > 0 && cluster_index >= uint(max_valid_clusters)) {
    return;
  }

  uvec4 meta = cluster_meta[ci];
  uint index_count = meta.y;
  if (index_count == 0u) {
    return;
  }

  uint vertex_offset = meta.x;
  uint vertex_count = meta.y;
  if (max_valid_vertices > 0 && vertex_offset + vertex_count > uint(max_valid_vertices)) {
    return; // Drop invalid command; avoid GPU memory fault
  }

  vec4 parent_sphere = cluster_parent_sphere[ci];
  vec4 sphere = cluster_sphere[ci];
  vec3 center = sphere.xyz;
  float radius = sphere.w;

  /* Coarse parent group sphere pre-culling + cluster sphere frustum culling
   * (Gribb-Hartmann planes, a*x + b*y + c*z + d >= 0 inside). */
  if (meta.w != 0xFFFFFFFFu && parent_sphere.w >= radius) {
    for (int p = 0; p < 6; ++p) {
      float dp = dot(frustum_planes[p].xyz, parent_sphere.xyz) + frustum_planes[p].w;
      if (dp < -(parent_sphere.w + culling_margin)) {
        return;
      }
    }
  }
  for (int p = 0; p < 6; ++p) {
    float d = dot(frustum_planes[p].xyz, center) + frustum_planes[p].w;
    if (d < -(radius + culling_margin)) {
      return;
    }
  }

  /* Tessera DAG Cut Criterion:
   * A cluster is rendered iff: E_cluster <= tau AND E_parent > tau
   *
   * NOTE: There is intentionally NO normal-cone backface rejection in the
   * traversal. Foliage and other open, non-watertight card geometry is
   * two-sided: a normal bounding cone on open cards has a spread close to
   * pi, which would prematurely cull clusters facing away from the camera
   * and leave only grazing silhouette edges visible. Occlusion is handled
   * by the depth pre-pass instead. */
  vec2 errs = cluster_errors[ci];
  /* Errors are stored in the SOURCE mesh's units (the .nc2 / archetype build
   * space). The instance matrix that baked this traversal transforms
   * positions into world space, but not the errors — apply the reference
   * instance's scale (`error_scale`) so the projected screen error matches
   * the real world-space silhouette (botanical archetype sources are often
   * authored in mm/cm; without this the cut over-refines by the unit ratio
   * and ignores the pixel-error threshold). */
  float err_c = errs.x * error_scale;
  float err_p = (errs.y >= 3.0e38) ? errs.y : (errs.y * error_scale);

  const float near_clip = 0.01;
  float dist_c = max(distance(camera_pos, center) - radius, near_clip);
  float screen_err_c = (err_c <= 0.0) ? 0.0 : (err_c * proj_scale_y * (viewport_height * 0.5) / dist_c);

  /* Early out: if cluster error exceeds pixel error, cluster is too coarse */
  if (screen_err_c > pixel_error) {
    return;
  }

  float dist_p = max(distance(camera_pos, parent_sphere.xyz) - parent_sphere.w, near_clip);
  float screen_err_p = (err_p >= 3.0e38) ? 1.0e39 : (err_p * proj_scale_y * (viewport_height * 0.5) / dist_p);

  /* Early out: if parent error is below or equal to pixel error, parent is fine enough */
  if (screen_err_p <= pixel_error) {
    return;
  }

  if ((max_valid_clusters > 0 && cluster_index >= uint(max_valid_clusters)) ||
      (max_valid_vertices > 0 && vertex_offset + vertex_count > uint(max_valid_vertices))) {
    return; // Drop invalid command; avoid GPU memory fault
  }

  if (use_elements_command > 0) {
    /* Bucketed Single Indirect Instanced Draw mode (`render_forward`):
     * every visible cluster is appended to the index-count bucket matching
     * its size. Per bucket b, `draw_counters[76 + b]` accumulates the visible
     * count and `draw_commands[b * cluster_count + slot]` stores the visible
     * cluster index. The finalize pass writes one
     * DrawArraysIndirectCommand per bucket at `draw_counters[84 + b*4]`
     * (vertexCount = bucket size, instanceCount = count * instances). Each
     * bucket region holds at most `cluster_count` entries (a bucket can never
     * overflow because every visible cluster lands in exactly one bucket). */
    uint b = bucket_of(index_count);
    uint slot = atomicAdd(draw_counters[76 + b], 1u);
    if (slot >= uint(cluster_count)) {
      return;
    }
    atomicAdd(draw_counters[0], 1u);
    if (record_telemetry != 0) {
      atomicAdd(draw_counters[64], 1u);
      atomicAdd(draw_counters[65], index_count / 3u);
      atomicMax(draw_counters[66], 65535u - min(meta.z, 65535u));
      atomicMax(draw_counters[67], meta.z);
    }
    draw_commands[b * uint(cluster_count) + slot] = ci;
  }
  else {
    /* Multi-Draw Indirect mode (Eevee / Visibility ID pass): */
    uint mat = 0u;
    if (max_materials > 1) {
      mat = cluster_materials[ci];
      if (mat >= uint(max_materials)) {
        mat = 0u;
      }
    }
    uint counter_val = draw_counters[mat];
    if ((max_clusters > 0) && (counter_val >= uint(max_clusters))) {
      return;
    }
    uint slot = atomicAdd(draw_counters[mat], 1u);
    if (slot >= uint(max_commands_per_material)) {
      return;
    }
    if (record_telemetry != 0) {
      atomicAdd(draw_counters[64], 1u);
      atomicAdd(draw_counters[65], index_count / 3u);
      atomicMax(draw_counters[66], 65535u - min(meta.z, 65535u));
      atomicMax(draw_counters[67], meta.z);
    }
    uint base = (mat * uint(max_commands_per_material) + slot) * 4u;
    draw_commands[base + 0u] = meta.y;
    draw_commands[base + 1u] = 1u;
    draw_commands[base + 2u] = meta.x;
    draw_commands[base + 3u] = ci;
  }
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
    /* binding 3: x = firstIndex, y = indexCount, z = cluster LOD level,
     * w = encoded parent index (0xFFFFFFFF for roots). */
    info.storage_buf(3, gshader::Qualifier::read, "uvec4", "cluster_meta[]");
    /* binding 4: indirect draw commands (4 or 5 uints per command). */
    info.storage_buf(4, gshader::Qualifier::write, "uint", "draw_commands[]");
    /* binding 5: shared atomic draw counters per material. */
    info.storage_buf(5, gshader::Qualifier::read_write, "uint", "draw_counters[]");
    /* binding 6: material index per cluster. */
    info.storage_buf(6, gshader::Qualifier::read, "uint", "cluster_materials[]");

    info.push_constant(gshader::Type::float3_t, "camera_pos");
    info.push_constant(gshader::Type::float_t, "viewport_height");
    info.push_constant(gshader::Type::float_t, "proj_scale_y");
    info.push_constant(gshader::Type::float_t, "pixel_error");
    info.push_constant(gshader::Type::float_t, "culling_margin");
    /* Reference instance scale applied to the source-unit cluster errors. */
    info.push_constant(gshader::Type::float_t, "error_scale");
    info.push_constant(gshader::Type::int_t, "cluster_count");
    info.push_constant(gshader::Type::int_t, "max_clusters");
    info.push_constant(gshader::Type::int_t, "max_commands");
    info.push_constant(gshader::Type::int_t, "max_materials");
    info.push_constant(gshader::Type::int_t, "max_commands_per_material");
    info.push_constant(gshader::Type::int_t, "record_telemetry");
    info.push_constant(gshader::Type::int_t, "max_valid_clusters");
    info.push_constant(gshader::Type::int_t, "max_valid_vertices");
    info.push_constant(gshader::Type::int_t, "use_elements_command");
    info.push_constant(gshader::Type::float4_t, "frustum_planes", 6);
    /* Elements mode: max cluster index count (sizes the last bucket). */
    info.push_constant(gshader::Type::int_t, "bucket_max");
    /* 1 when this dispatch is the args-finalize pass (cluster_count == 0). */
    info.push_constant(gshader::Type::int_t, "finalize_pass");
    /* Object-instance count baked into the per-bucket instanceCounts. */
    info.push_constant(gshader::Type::int_t, "finalize_multiplier");

    /* Runtime-created shader: register the raw source as "generated" content
     * (same mechanism as the `gpu` Python module). */
    info.compute_source("tessera_traverse.glsl");
    info.compute_source_generated = COMPUTE_SOURCE;

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

gpu::Shader *createMultiInstanceTraverseShader()
{
    namespace gshader = blender::gpu::shader;

    static const char *MULTI_INSTANCE_SOURCE = R"(
void main()
{
  uint ci = uint(gl_GlobalInvocationID.x);
  uint inst_id = uint(gl_GlobalInvocationID.y);
  if (ci >= uint(cluster_count) || inst_id >= uint(instance_count)) {
    return;
  }

  InstanceData inst = instances[inst_id];

  /* Stage 1: Instance bounding sphere culling against the frustum. */
  vec3 inst_center = (inst.model_matrix * vec4(inst.bounding_sphere.xyz, 1.0)).xyz;
  float scale_x = length(inst.model_matrix[0].xyz);
  float scale_y = length(inst.model_matrix[1].xyz);
  float scale_z = length(inst.model_matrix[2].xyz);
  float max_scale = max(scale_x, max(scale_y, scale_z));
  float inst_radius = inst.bounding_sphere.w * max_scale;

  for (int p = 0; p < 6; ++p) {
    float d = dot(frustum_planes[p].xyz, inst_center) + frustum_planes[p].w;
    if (d < -(inst_radius + culling_margin)) {
      return; /* Instance is outside frustum */
    }
  }

  uint cluster_index = ci;
  if (max_valid_clusters > 0 && cluster_index >= uint(max_valid_clusters)) {
    return;
  }

  /* Stage 2: Cluster DAG LOD cut evaluation on surviving instances. */
  uvec4 meta = cluster_meta[ci];
  uint index_count = meta.y;
  if (index_count == 0u) {
    return;
  }

  uint vertex_offset = meta.x;
  uint vertex_count = meta.y;
  if (max_valid_vertices > 0 && vertex_offset + vertex_count > uint(max_valid_vertices)) {
    return; // Drop invalid command; avoid GPU memory fault
  }

  vec4 sphere = cluster_sphere[ci];
  vec3 center = (inst.model_matrix * vec4(sphere.xyz, 1.0)).xyz;
  float radius = sphere.w * max_scale;

  for (int p = 0; p < 6; ++p) {
    float d = dot(frustum_planes[p].xyz, center) + frustum_planes[p].w;
    if (d < -(radius + culling_margin)) {
      return;
    }
  }

  vec2 errs = cluster_errors[ci];
  float err_c = errs.x * max_scale;
  float err_p = errs.y * max_scale;

  const float near_clip = 0.01;
  float dist_c = max(distance(camera_pos, center) - radius, near_clip);
  float screen_err_c = (err_c <= 0.0) ? 0.0 : (err_c * proj_scale_y * (viewport_height * 0.5) / dist_c);

  /* Early out: if cluster error exceeds pixel error, cluster is too coarse */
  if (screen_err_c > pixel_error) {
    return;
  }

  vec4 parent_sphere = cluster_parent_sphere[ci];
  vec3 parent_center = (inst.model_matrix * vec4(parent_sphere.xyz, 1.0)).xyz;
  float parent_radius = parent_sphere.w * max_scale;
  float dist_p = max(distance(camera_pos, parent_center) - parent_radius, near_clip);
  float screen_err_p = (err_p >= 3.0e38) ? 1.0e39 : (err_p * proj_scale_y * (viewport_height * 0.5) / dist_p);

  /* Early out: if parent error is below or equal to pixel error, parent is fine enough */
  if (screen_err_p <= pixel_error) {
    return;
  }

  uint mat = 0u;
  if (max_materials > 1) {
    mat = cluster_materials[ci];
    if (mat >= uint(max_materials)) {
      mat = 0u;
    }
  }

  if ((max_valid_clusters > 0 && cluster_index >= uint(max_valid_clusters)) ||
      (max_valid_vertices > 0 && vertex_offset + vertex_count > uint(max_valid_vertices))) {
    return; // Drop invalid command; avoid GPU memory fault
  }

  uint counter_val = draw_counters[mat];
  if ((max_clusters > 0) && (counter_val >= uint(max_clusters))) {
    return;
  }
  uint slot = atomicAdd(draw_counters[mat], 1u);
  if (slot >= uint(max_commands_per_material)) {
    return;
  }

  if (record_telemetry != 0) {
    atomicAdd(draw_counters[64], 1u);
    atomicAdd(draw_counters[65], index_count / 3u);
    atomicMin(draw_counters[66], meta.z);
    atomicMax(draw_counters[67], meta.z);
  }

  /* DrawArraysIndirectCommand layout (4 x uint32): count, instanceCount, first, baseInstance */
  uint base = (mat * uint(max_commands_per_material) + slot) * 4u;
  draw_commands[base + 0u] = meta.y;
  draw_commands[base + 1u] = 1u;
  draw_commands[base + 2u] = meta.x;
  draw_commands[base + 3u] = inst_id;
}
)";

    gshader::ShaderCreateInfo info("pyGPU_Shader");
    info.local_group_size(64, 1, 1);
    info.builtins_ |= gshader::BuiltinBits::NO_BUFFER_TYPE_LINTING;

    info.typedef_source_generated =
        "struct InstanceData {\n"
        "  mat4 model_matrix;\n"
        "  vec4 bounding_sphere;\n"
        "  uint object_id;\n"
        "  uint mesh_dag_root;\n"
        "  uvec2 _pad;\n"
        "};\n";
    info.typedef_source("gpu_shader_python_typedef_lib.glsl");

    /* binding 0: bounding spheres */
    info.storage_buf(0, gshader::Qualifier::read, "vec4", "cluster_sphere[]");
    /* binding 1: parent bounding spheres */
    info.storage_buf(1, gshader::Qualifier::read, "vec4", "cluster_parent_sphere[]");
    /* binding 2: cluster errors */
    info.storage_buf(2, gshader::Qualifier::read, "vec2", "cluster_errors[]");
    /* binding 3: cluster meta */
    info.storage_buf(3, gshader::Qualifier::read, "uvec4", "cluster_meta[]");
    /* binding 4: instance buffer (SSBO) */
    info.storage_buf(4, gshader::Qualifier::read, "InstanceData", "instances[]");
    /* binding 5: indirect draw commands */
    info.storage_buf(5, gshader::Qualifier::write, "uint", "draw_commands[]");
    /* binding 6: draw counters */
    info.storage_buf(6, gshader::Qualifier::read_write, "uint", "draw_counters[]");
    /* binding 7: materials */
    info.storage_buf(7, gshader::Qualifier::read, "uint", "cluster_materials[]");

    info.push_constant(gshader::Type::float3_t, "camera_pos");
    info.push_constant(gshader::Type::float_t, "viewport_height");
    info.push_constant(gshader::Type::float_t, "proj_scale_y");
    info.push_constant(gshader::Type::float_t, "pixel_error");
    info.push_constant(gshader::Type::float_t, "culling_margin");
    info.push_constant(gshader::Type::int_t, "cluster_count");
    info.push_constant(gshader::Type::int_t, "instance_count");
    info.push_constant(gshader::Type::int_t, "max_clusters");
    info.push_constant(gshader::Type::int_t, "max_commands");
    info.push_constant(gshader::Type::int_t, "max_materials");
    info.push_constant(gshader::Type::int_t, "max_commands_per_material");
    info.push_constant(gshader::Type::int_t, "record_telemetry");
    info.push_constant(gshader::Type::int_t, "max_valid_clusters");
    info.push_constant(gshader::Type::int_t, "max_valid_vertices");
    info.push_constant(gshader::Type::float4_t, "frustum_planes", 6);

    info.compute_source("tessera_multi_instance_traverse.glsl");
    info.compute_source_generated = MULTI_INSTANCE_SOURCE;

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

}  // namespace tessera
}  // namespace blender
