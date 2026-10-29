/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 2: Visibility buffer rendering implementation.
 */

#include "tessera_visibility.hh"

#include "tessera_lod.hh"

#include "GPU_batch.hh"
#include "GPU_capabilities.hh"
#include "GPU_context.hh"
#include "GPU_index_buffer.hh"
#include "GPU_shader_create_info.hh"
#include "GPU_state.hh"
#include "GPU_vertex_buffer.hh"
#include "GPU_vertex_format.hh"

#include <BLI_math_matrix.hh>

#include <cstdio>
#include <cstring>

namespace blender {
namespace tessera {

static float3x3 compute_normal_matrix(const float4x4 &model_mat)
{
    float3x3 m3(model_mat);
    bool success = false;
    float3x3 inv = math::invert(m3, success);
    if (!success) {
        return float3x3::identity();
    }
    return math::transpose(inv);
}

namespace {

/** True when a real GPU backend is active. */
bool gpu_available()
{
    return GPU_context_active_get() != nullptr &&
           GPU_backend_get_type() != GPU_BACKEND_NONE;
}

/* ---------------------------------------------------------------------------
 * Visibility pass shaders.
 *
 * The vertex shader pulls vertex data from SSBOs (no vertex attributes are
 * consumed): gl_VertexID is the absolute index-buffer position, which also
 * identifies the global triangle (gl_VertexID / 3). The one-hot corner
 * indicator is interpolated perspective-correctly by the hardware, yielding
 * exactly the barycentric weights used for attribute interpolation.
 * ------------------------------------------------------------------------- */

static const char *VIS_VERT_SOURCE = R"(
void main()
{
  uint global_idx = vid;
  uint corner = global_idx % 3u;
  uint idx = tri_indices[global_idx];
  vec3 pos = vec3(positions[idx * 3u], positions[idx * 3u + 1u],
                  positions[idx * 3u + 2u]);
  gl_Position = view_proj * vec4(pos, 1.0);

  bary = vec2(corner == 0u ? 1.0 : 0.0, corner == 1u ? 1.0 : 0.0);
  tri_id = (global_idx / 3u) + 1u;

  /* Zero-search O(1) direct cluster lookup via gl_BaseInstanceARB (gpu_BaseInstance) */
  cluster_id = uint(gpu_BaseInstance);
}
)";

static const char *VIS_FRAG_SOURCE = R"(
void main()
{
  out_vis = uvec4(tri_id, packHalf2x16(bary), cluster_id, gl_FrontFacing ? 1u : 0u);
}
)";

/* ---------------------------------------------------------------------------
 * Material pass: fetch the visibility record, reconstruct the attributes and
 * shade. The background is encoded with tri_id == 0 or 0xFFFFFFFF.
 * ------------------------------------------------------------------------- */

static const char *MATERIAL_VERT_SOURCE = R"(
void main()
{
  gl_Position = vec4(pos.xy, 1.0, 1.0);
}
)";

static const char *MATERIAL_FRAG_SOURCE = R"(
void main()
{
  ivec2 px = ivec2(gl_FragCoord.xy);
  uvec4 vis = texelFetch(vis_texture, px, 0);
  if (vis.x == 0u || vis.x == 0xFFFFFFFFu) {
    /* No Tessera geometry here: discard immediately to avoid running expensive
     * fragment math, texture fetches, and depth reconstruction. */
    discard;
  }
  uint tri = vis.x - 1u;
  float depth_val = texelFetch(depth_texture, px, 0).r;
  gl_FragDepth = depth_val;
  vec2 bary_xy = unpackHalf2x16(vis.y);
  vec3 bary = vec3(bary_xy.x, bary_xy.y, 1.0 - bary_xy.x - bary_xy.y);

  uint i0 = tri_indices[tri * 3u + 0u];
  uint i1 = tri_indices[tri * 3u + 1u];
  uint i2 = tri_indices[tri * 3u + 2u];

  vec3 p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
  vec3 p1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]);
  vec3 p2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]);

  vec3 n0 = vec3(normals[i0 * 3u], normals[i0 * 3u + 1u], normals[i0 * 3u + 2u]);
  vec3 n1 = vec3(normals[i1 * 3u], normals[i1 * 3u + 1u], normals[i1 * 3u + 2u]);
  vec3 n2 = vec3(normals[i2 * 3u], normals[i2 * 3u + 1u], normals[i2 * 3u + 2u]);

  vec2 u0 = vec2(uvs[i0 * 2u], uvs[i0 * 2u + 1u]);
  vec2 u1 = vec2(uvs[i1 * 2u], uvs[i1 * 2u + 1u]);
  vec2 u2 = vec2(uvs[i2 * 2u], uvs[i2 * 2u + 1u]);

  vec4 t0 = vec4(tangents[i0 * 4u], tangents[i0 * 4u + 1u], tangents[i0 * 4u + 2u], tangents[i0 * 4u + 3u]);
  vec4 t1 = vec4(tangents[i1 * 4u], tangents[i1 * 4u + 1u], tangents[i1 * 4u + 2u], tangents[i1 * 4u + 3u]);
  vec4 t2 = vec4(tangents[i2 * 4u], tangents[i2 * 4u + 1u], tangents[i2 * 4u + 2u], tangents[i2 * 4u + 3u]);

  vec3 pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
  vec3 nor_v = bary.x * n0 + bary.y * n1 + bary.z * n2;
  float nor_len = length(nor_v);
  vec3 nor = nor_len > 1e-20 ? nor_v / nor_len : vec3(0.0, 0.0, 1.0);
  vec2 uv = bary.x * u0 + bary.y * u1 + bary.z * u2;
  vec4 tan_v = bary.x * t0 + bary.y * t1 + bary.z * t2;

  float out_alpha = (reverse_z != 0) ? 0.0 : 1.0;

  if (shading_mode == 1u) {
    /* Cluster visualization */
    uint cid = vis.z;
    float hue = fract(float((cid * 2654435761u) & 0xFFFFFFu) / float(0xFFFFFFu) +
                      0.03125);
    float val = 0.45 + 0.55 * (float((tri * 1973u + cid * 7919u) & 0xFFFFu) / 65535.0);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(hue) + K.xyz) * 6.0 - K.www);
    vec3 rgb = mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), 0.85);
    out_color = vec4(rgb * val, out_alpha);
    return;
  }
  if (shading_mode == 2u) {
    vec3 disp_nor = (vis.w != 0u) ? nor : -nor;
    out_color = vec4(disp_nor * 0.5 + 0.5, out_alpha);
    return;
  }
  if (shading_mode == 3u) {
    float t_len = length(tan_v.xyz);
    vec3 t_norm = (t_len > 1e-6) ? (tan_v.xyz / t_len) : vec3(1.0, 0.0, 0.0);
    out_color = vec4(t_norm * 0.5 + 0.5, out_alpha);
    return;
  }
  if (shading_mode == 4u) {
    out_color = vec4(fract(uv.x), fract(uv.y), 0.0, out_alpha);
    return;
  }

  /* Shading Mode 0: full material texturing + normal mapping */
  uint mat_idx = tri_materials[tri];
  if (mat_idx >= 4u) {
    mat_idx = 0u;
  }

  vec4 base_col = material_params[mat_idx * 4u + 0u];
  vec4 props = material_params[mat_idx * 4u + 1u];
  vec4 rough_params = material_params[mat_idx * 4u + 2u];
  vec4 extra_params = material_params[mat_idx * 4u + 3u];
  bool has_d = (props.x > 0.5);
  bool has_n = (props.y > 0.5);
  bool has_r = (props.z > 0.5);
  float spec_level = props.w;
  float metallic = base_col.w;
  float is_srgb = extra_params.x;
  float has_spec_tint = extra_params.y;

  float rough_val = rough_params.x;
  float rough_gamma = rough_params.y;
  float rough_scale = rough_params.z;
  float rough_offset = rough_params.w;

  vec4 tex_diff = vec4(1.0);
  vec3 tex_norm = vec3(0.5, 0.5, 1.0);
  float tex_rough = rough_val;

  vec2 ddx_uv = dFdx(uv);
  vec2 ddy_uv = dFdy(uv);
  float d_tri_x = dFdx(float(tri));
  float d_tri_y = dFdy(float(tri));
  if (abs(d_tri_x) > 0.5) {
    ddx_uv = vec2(0.0);
  }
  if (abs(d_tri_y) > 0.5) {
    ddy_uv = vec2(0.0);
  }
  const float max_grad = 0.05;
  ddx_uv = clamp(ddx_uv, vec2(-max_grad), vec2(max_grad)) * tex_scale_inv;
  ddy_uv = clamp(ddy_uv, vec2(-max_grad), vec2(max_grad)) * tex_scale_inv;

  if (mat_idx == 0u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_0, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_0, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_0, uv, ddx_uv, ddy_uv).r;
  }
  else if (mat_idx == 1u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_1, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_1, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_1, uv, ddx_uv, ddy_uv).r;
  }
  else if (mat_idx == 2u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_2, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_2, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_2, uv, ddx_uv, ddy_uv).r;
  }
  else {
    if (has_d) tex_diff = textureGrad(tex_diffuse_3, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_3, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_3, uv, ddx_uv, ddy_uv).r;
  }

  if (has_r) {
    if (abs(rough_gamma - 1.0) > 1e-4) {
      tex_rough = pow(max(tex_rough, 0.0), rough_gamma);
    }
    tex_rough = tex_rough * rough_scale + rough_offset;
  }

  vec4 albedo = has_d ? (tex_diff * vec4(base_col.rgb, 1.0)) : vec4(base_col.rgb, 1.0);
  float roughness = clamp(has_r ? tex_rough : rough_val, 0.001, 1.0);

  /* Alpha cutout: reject transparent texel borders of foliage cards so the
   * shaded silhouette matches the leaf texture instead of the full quad. */
  if (has_d && tex_diff.a < 0.33) {
    discard;
  }

  /* MikkTSpace normal perturbation in world space using vertex normal and 4D tangent */
  mat3 norm_mat = normal_mat;
  vec3 world_nor = normalize(norm_mat * nor);

  vec3 m_tan = mat3(model_mat) * tan_v.xyz;
  float m_tan_len = length(m_tan);
  vec3 world_tan = (m_tan_len > 1e-6) ? (m_tan / m_tan_len) : vec3(1.0, 0.0, 0.0);

  vec3 N = world_nor;
  vec3 T = world_tan - N * dot(N, world_tan);
  float t_len = length(T);
  T = (t_len > 1e-6) ? (T / t_len) : vec3(1.0, 0.0, 0.0);

  float handedness = (tan_v.w < 0.0) ? -1.0 : 1.0;
  float det_sign = (determinant(mat3(model_mat)) < 0.0) ? -1.0 : 1.0;
  vec3 B = cross(N, T) * (handedness * det_sign);
  mat3 TBN = mat3(T, B, N);

  vec3 n_mapped = tex_norm * 2.0 - 1.0;
  vec3 shaded_nor = has_n ? normalize(TBN * n_mapped) : N;

  bool is_front_facing = (vis.w != 0u);
  if (!is_front_facing) {
    shaded_nor = -shaded_nor;
  }

  /* Headlight / camera lighting with specular highlight */
  vec3 world_pos = (model_mat * vec4(pos, 1.0)).xyz;
  vec3 light_dir = normalize(camera_pos - world_pos);
  float NdotL = clamp(dot(shaded_nor, light_dir), 0.0, 1.0);

  vec3 view_dir = light_dir;
  vec3 half_dir = normalize(light_dir + view_dir);
  float NdotH = clamp(dot(shaded_nor, half_dir), 0.0, 1.0);
  float spec_power = (1.0 - clamp(roughness, 0.04, 0.98)) * 64.0;
  float spec = pow(NdotH, max(1.0, spec_power)) * (1.0 - roughness);

  vec3 diff_term = albedo.rgb * (0.15 + 0.85 * NdotL);
  vec3 spec_term = vec3(spec * 0.3 * spec_level);
  if (has_spec_tint > 0.5) {
    spec_term *= albedo.rgb;
  }
  vec3 final_rgb = diff_term + spec_term;

  out_color = vec4(final_rgb, out_alpha);
}
)";

/* ---------------------------------------------------------------------------
 * G-Buffer pass: output EEVEE deferred G-Buffer targets:
 * Attachment 0: combined_tx (radiance / emission)
 * Attachment 1: header_tx (UInt32 closure type modes and flags)
 * Attachment 2: normal_tx (UNorm16x2 octahedral packed world normal)
 * Attachment 3: closure_tx layer 0 (Closure 0 Diffuse color_pack albedo)
 * Attachment 4: closure_tx layer 1 (Closure 1 ReflectionColorless roughness & specular)
 * Depth: depth_tx (Reverse-Z depth)
 * ------------------------------------------------------------------------- */

static const char *GBUFFER_FRAG_SOURCE = R"(
vec2 normal_pack(vec3 N)
{
  N /= (abs(N.x) + abs(N.y) + abs(N.z));
  vec2 s = sign(N.xy);
  s.x = (s.x == 0.0) ? 1.0 : s.x;
  s.y = (s.y == 0.0) ? 1.0 : s.y;
  N.xy = (N.z >= 0.0) ? N.xy : ((1.0 - abs(N.yx)) * s);
  return N.xy * 0.5 + 0.5;
}

vec4 closure_color_pack(vec3 color)
{
  float max_comp = max(color.x, max(color.y, color.z));
  float exponent = (max_comp > 1.0) ? ((max_comp > 2.0) ? ((max_comp > 4.0) ? 3.0 : 2.0) : 1.0) : 0.0;
  return vec4(color / exp2(exponent), exponent / 3.0);
}

uint geometry_normal_pack(vec3 Ng, vec3 N)
{
  const float quantization_multiplier = 1.360;
  vec3 Ng_quantize = normalize(round(quantization_multiplier * Ng));
  if (dot(N, Ng) > dot(Ng, Ng_quantize)) {
    return 0u;
  }
  uint data = 0u;
  data |= (Ng_quantize.x > 0.0) ? (1u << 0u) : 0u;
  data |= (Ng_quantize.y > 0.0) ? (1u << 1u) : 0u;
  data |= (Ng_quantize.z > 0.0) ? (1u << 2u) : 0u;
  data |= (Ng_quantize.x < 0.0) ? (1u << 3u) : 0u;
  data |= (Ng_quantize.y < 0.0) ? (1u << 4u) : 0u;
  data |= (Ng_quantize.z < 0.0) ? (1u << 5u) : 0u;
  return data << 20u;
}

void main()
{
  ivec2 px = ivec2(gl_FragCoord.xy);
  uvec4 vis = texelFetch(vis_texture, px, 0);
  if (vis.x == 0u || vis.x == 0xFFFFFFFFu) {
    discard;
  }
  uint tri = vis.x - 1u;
  float depth_val = texelFetch(depth_texture, px, 0).r;
  gl_FragDepth = depth_val;
  vec2 bary_xy = unpackHalf2x16(vis.y);
  vec3 bary = vec3(bary_xy.x, bary_xy.y, 1.0 - bary_xy.x - bary_xy.y);

  uint i0 = tri_indices[tri * 3u + 0u];
  uint i1 = tri_indices[tri * 3u + 1u];
  uint i2 = tri_indices[tri * 3u + 2u];

  vec3 p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
  vec3 p1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]);
  vec3 p2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]);

  vec3 n0 = vec3(normals[i0 * 3u], normals[i0 * 3u + 1u], normals[i0 * 3u + 2u]);
  vec3 n1 = vec3(normals[i1 * 3u], normals[i1 * 3u + 1u], normals[i1 * 3u + 2u]);
  vec3 n2 = vec3(normals[i2 * 3u], normals[i2 * 3u + 1u], normals[i2 * 3u + 2u]);

  vec2 u0 = vec2(uvs[i0 * 2u], uvs[i0 * 2u + 1u]);
  vec2 u1 = vec2(uvs[i1 * 2u], uvs[i1 * 2u + 1u]);
  vec2 u2 = vec2(uvs[i2 * 2u], uvs[i2 * 2u + 1u]);

  vec4 t0 = vec4(tangents[i0 * 4u], tangents[i0 * 4u + 1u], tangents[i0 * 4u + 2u], tangents[i0 * 4u + 3u]);
  vec4 t1 = vec4(tangents[i1 * 4u], tangents[i1 * 4u + 1u], tangents[i1 * 4u + 2u], tangents[i1 * 4u + 3u]);
  vec4 t2 = vec4(tangents[i2 * 4u], tangents[i2 * 4u + 1u], tangents[i2 * 4u + 2u], tangents[i2 * 4u + 3u]);

  vec3 nor_v = bary.x * n0 + bary.y * n1 + bary.z * n2;
  float nor_len = length(nor_v);
  vec3 nor = nor_len > 1e-20 ? nor_v / nor_len : vec3(0.0, 0.0, 1.0);
  vec2 uv = bary.x * u0 + bary.y * u1 + bary.z * u2;
  vec4 tan_v = bary.x * t0 + bary.y * t1 + bary.z * t2;

  uint mat_idx = tri_materials[tri];
  if (mat_idx >= 4u) {
    mat_idx = 0u;
  }

  vec4 base_col = material_params[mat_idx * 4u + 0u];
  vec4 props = material_params[mat_idx * 4u + 1u];
  vec4 rough_params = material_params[mat_idx * 4u + 2u];
  vec4 extra_params = material_params[mat_idx * 4u + 3u];
  bool has_d = (props.x > 0.5);
  bool has_n = (props.y > 0.5);
  bool has_r = (props.z > 0.5);
  float spec_level = props.w;
  float metallic = base_col.w;
  float is_srgb = extra_params.x;
  float has_spec_tint = extra_params.y;

  float rough_val = rough_params.x;
  float rough_gamma = rough_params.y;
  float rough_scale = rough_params.z;
  float rough_offset = rough_params.w;

  vec4 tex_diff = vec4(1.0);
  vec3 tex_norm = vec3(0.5, 0.5, 1.0);
  float tex_rough = rough_val;

  vec2 ddx_uv = dFdx(uv);
  vec2 ddy_uv = dFdy(uv);
  float d_tri_x = dFdx(float(tri));
  float d_tri_y = dFdy(float(tri));
  if (abs(d_tri_x) > 0.5) {
    ddx_uv = vec2(0.0);
  }
  if (abs(d_tri_y) > 0.5) {
    ddy_uv = vec2(0.0);
  }
  const float max_grad = 0.05;
  ddx_uv = clamp(ddx_uv, vec2(-max_grad), vec2(max_grad)) * tex_scale_inv;
  ddy_uv = clamp(ddy_uv, vec2(-max_grad), vec2(max_grad)) * tex_scale_inv;

  if (mat_idx == 0u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_0, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_0, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_0, uv, ddx_uv, ddy_uv).r;
  }
  else if (mat_idx == 1u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_1, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_1, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_1, uv, ddx_uv, ddy_uv).r;
  }
  else if (mat_idx == 2u) {
    if (has_d) tex_diff = textureGrad(tex_diffuse_2, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_2, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_2, uv, ddx_uv, ddy_uv).r;
  }
  else {
    if (has_d) tex_diff = textureGrad(tex_diffuse_3, uv, ddx_uv, ddy_uv);
    if (has_n) tex_norm = textureGrad(tex_normal_3, uv, ddx_uv, ddy_uv).xyz;
    if (has_r) tex_rough = textureGrad(tex_rough_3, uv, ddx_uv, ddy_uv).r;
  }

  if (has_r) {
    if (abs(rough_gamma - 1.0) > 1e-4) {
      tex_rough = pow(max(tex_rough, 0.0), rough_gamma);
    }
    tex_rough = tex_rough * rough_scale + rough_offset;
  }

  vec4 albedo = has_d ? (tex_diff * vec4(base_col.rgb, 1.0)) : vec4(base_col.rgb, 1.0);
  float roughness = clamp(has_r ? tex_rough : rough_val, 0.001, 1.0);

  /* Alpha cutout: reject transparent texel borders of foliage cards so the
   * deferred G-Buffer silhouette matches the leaf texture. */
  if (has_d && tex_diff.a < 0.33) {
    discard;
  }

  mat3 norm_mat = normal_mat;
  vec3 world_nor = normalize(norm_mat * nor);

  vec3 m_tan = mat3(model_mat) * tan_v.xyz;
  float m_tan_len = length(m_tan);
  vec3 world_tan = (m_tan_len > 1e-6) ? (m_tan / m_tan_len) : vec3(1.0, 0.0, 0.0);

  vec3 N = world_nor;
  vec3 T = world_tan - N * dot(N, world_tan);
  float t_len = length(T);
  T = (t_len > 1e-6) ? (T / t_len) : vec3(1.0, 0.0, 0.0);

  float handedness = (tan_v.w < 0.0) ? -1.0 : 1.0;
  float det_sign = (determinant(mat3(model_mat)) < 0.0) ? -1.0 : 1.0;
  vec3 B = cross(N, T) * (handedness * det_sign);
  mat3 TBN = mat3(T, B, N);

  vec3 n_mapped = tex_norm * 2.0 - 1.0;
  vec3 shaded_nor = has_n ? normalize(TBN * n_mapped) : N;

  /* True geometric normal of the triangle in world space for EEVEE shadow/grazing biasing */
  vec3 world_p0 = (model_mat * vec4(p0, 1.0)).xyz;
  vec3 world_p1 = (model_mat * vec4(p1, 1.0)).xyz;
  vec3 world_p2 = (model_mat * vec4(p2, 1.0)).xyz;
  vec3 world_geo_n = cross(world_p1 - world_p0, world_p2 - world_p0);
  float geo_len = length(world_geo_n);
  world_geo_n = (geo_len > 1e-12) ? (world_geo_n / geo_len) : world_nor;

  /* Preserve true winding orientation recorded in vis.w */
  bool is_front_facing = (vis.w != 0u);
  if (!is_front_facing) {
    world_geo_n = -world_geo_n;
    shaded_nor = -shaded_nor;
  }

  uint geo_norm_bits = geometry_normal_pack(world_geo_n, shaded_nor);
  uint header = 0x31u | geo_norm_bits;

  float f0 = mix(0.08 * spec_level, 1.0, metallic);
  if (has_spec_tint > 0.5) {
    f0 *= clamp(dot(albedo.rgb, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
  }

  out_radiance = vec4(0.0);
  out_header = header;
  out_normal = normal_pack(shaded_nor);
  out_closure1 = closure_color_pack(albedo.rgb);
  out_closure2 = vec4(roughness, 0.0, f0, 0.0);
}
)";

/* Depth-only prepass shaders (vertex pulling). */
static const char *DEPTH_VERT_SOURCE = R"(
void main()
{
  uint global_idx = vid;
  uint idx = tri_indices[global_idx];
  vec3 pos = vec3(positions[idx * 3u], positions[idx * 3u + 1u],
                  positions[idx * 3u + 2u]);
  gl_Position = view_proj * vec4(pos, 1.0);

  uint mat_idx = tri_materials[global_idx / 3u];
  if (mat_idx >= 4u) {
    mat_idx = 0u;
  }
  v_mat = mat_idx;
  v_uv = vec2(uvs[idx * 2u], uvs[idx * 2u + 1u]);
}
)";

static const char *DEPTH_FRAG_SOURCE = R"(
void main()
{
  /* Alpha cutout: leaf / grass cards carry their silhouette in the base
   * color texture alpha channel. Reject transparent texel borders so the
   * depth pre-pass does not leave rectangular halos around foliage. */
  if (has_alpha_mask != 0) {
    float slot_has_d = (v_mat == 0u) ? has_tex_flags.x :
                       (v_mat == 1u) ? has_tex_flags.y :
                       (v_mat == 2u) ? has_tex_flags.z : has_tex_flags.w;
    if (slot_has_d > 0.5) {
      vec4 tex_col = (v_mat == 0u) ? texture(tex_diffuse_0, v_uv) :
                     (v_mat == 1u) ? texture(tex_diffuse_1, v_uv) :
                     (v_mat == 2u) ? texture(tex_diffuse_2, v_uv) :
                                     texture(tex_diffuse_3, v_uv);
      if (tex_col.a < 0.33) {
        discard;
      }
    }
  }
}
)";

/** Create the visibility-pass shader (vertex pulling + image store). */
gpu::Shader *createVisShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("pyGPU_Shader");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;
    info.builtins_ |= BuiltinBits::CLIP_CONTROL;

    info.early_fragment_test(true);
    info.depth_write(DepthWrite::UNCHANGED);
    info.vertex_source("tessera_vis");
    info.vertex_source_generated = VIS_VERT_SOURCE;
    info.fragment_source("tessera_vis");
    info.fragment_source_generated = VIS_FRAG_SOURCE;

    /* Vertex pulling: vid attribute supplies the absolute index position. */
    info.vertex_in(0, Type::uint_t, "vid");
    info.fragment_out(0, Type::uint4_t, "out_vis");
    info.storage_buf(0, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(1, Qualifier::read, "float", "positions[]");

    info.push_constant(Type::float4x4_t, "view_proj");
    info.push_constant(Type::int_t, "debug_clusters");
    info.push_constant(Type::int_t, "cluster_count");
    info.push_constant(Type::int_t, "reverse_z");

    StageInterfaceInfo iface("tessera_vis_iface", "");
    iface.flat(Type::uint_t, "tri_id");
    iface.smooth(Type::float2_t, "bary");
    iface.flat(Type::uint_t, "cluster_id");
    info.vertex_out(iface);

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

/** Create the material-pass shader (fullscreen shading). */
gpu::Shader *createMaterialShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("pyGPU_Shader");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;
    info.builtins_ |= BuiltinBits::CLIP_CONTROL;

    info.vertex_source("tessera_material");
    info.vertex_source_generated = MATERIAL_VERT_SOURCE;
    info.fragment_source("tessera_material");
    info.fragment_source_generated = MATERIAL_FRAG_SOURCE;

    info.vertex_in(0, Type::float3_t, "pos");
    info.fragment_out(0, Type::float4_t, "out_color");

    info.depth_write(DepthWrite::ANY);
    info.storage_buf(0, Qualifier::read, "float", "positions[]");
    info.storage_buf(1, Qualifier::read, "float", "normals[]");
    info.storage_buf(2, Qualifier::read, "float", "uvs[]");
    info.storage_buf(3, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(4, Qualifier::read, "float", "tangents[]");
    info.storage_buf(5, Qualifier::read, "uint", "tri_materials[]");
    info.storage_buf(6, Qualifier::read, "vec4", "material_params[]");
    info.sampler(0, ImageType::Uint2D, "vis_texture");
    info.sampler(1, ImageType::Float2D, "depth_texture");

    info.sampler(2, ImageType::Float2D, "tex_diffuse_0");
    info.sampler(3, ImageType::Float2D, "tex_normal_0");
    info.sampler(4, ImageType::Float2D, "tex_rough_0");

    info.sampler(5, ImageType::Float2D, "tex_diffuse_1");
    info.sampler(6, ImageType::Float2D, "tex_normal_1");
    info.sampler(7, ImageType::Float2D, "tex_rough_1");

    info.sampler(8, ImageType::Float2D, "tex_diffuse_2");
    info.sampler(9, ImageType::Float2D, "tex_normal_2");
    info.sampler(10, ImageType::Float2D, "tex_rough_2");

    info.sampler(11, ImageType::Float2D, "tex_diffuse_3");
    info.sampler(12, ImageType::Float2D, "tex_normal_3");
    info.sampler(13, ImageType::Float2D, "tex_rough_3");

    info.push_constant(Type::float4x4_t, "model_mat");
    info.push_constant(Type::float3x3_t, "normal_mat");
    info.push_constant(Type::float3_t, "camera_pos");
    info.push_constant(Type::int_t, "shading_mode");
    info.push_constant(Type::int_t, "reverse_z");
    info.push_constant(Type::float_t, "tex_scale_inv");

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

/** Create the depth-only prepass shader (vertex pulling). */
gpu::Shader *createDepthPrepassShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("pyGPU_Shader");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;
    info.builtins_ |= BuiltinBits::CLIP_CONTROL;

    info.vertex_source("tessera_depth_prepass");
    info.vertex_source_generated = DEPTH_VERT_SOURCE;
    info.fragment_source("tessera_depth_prepass");
    info.fragment_source_generated = DEPTH_FRAG_SOURCE;

    info.vertex_in(0, Type::uint_t, "vid");
    info.storage_buf(0, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(1, Qualifier::read, "float", "positions[]");
    info.storage_buf(2, Qualifier::read, "float", "uvs[]");
    info.storage_buf(3, Qualifier::read, "uint", "tri_materials[]");

    info.sampler(0, ImageType::Float2D, "tex_diffuse_0");
    info.sampler(1, ImageType::Float2D, "tex_diffuse_1");
    info.sampler(2, ImageType::Float2D, "tex_diffuse_2");
    info.sampler(3, ImageType::Float2D, "tex_diffuse_3");

    info.push_constant(Type::float4x4_t, "view_proj");
    info.push_constant(Type::int_t, "reverse_z");
    info.push_constant(Type::int_t, "has_alpha_mask");
    info.push_constant(Type::float4_t, "has_tex_flags");

    StageInterfaceInfo iface("tessera_depth_iface", "");
    iface.smooth(Type::float2_t, "v_uv");
    iface.flat(Type::uint_t, "v_mat");
    info.vertex_out(iface);

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

/** Create the G-Buffer pass shader (fullscreen MRT output). */
gpu::Shader *createGbufferShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("pyGPU_Shader");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;
    info.builtins_ |= BuiltinBits::CLIP_CONTROL;

    info.vertex_source("tessera_gbuffer");
    info.vertex_source_generated = MATERIAL_VERT_SOURCE;
    info.fragment_source("tessera_gbuffer");
    info.fragment_source_generated = GBUFFER_FRAG_SOURCE;

    info.vertex_in(0, Type::float3_t, "pos");
    info.fragment_out(0, Type::float4_t, "out_radiance");
    info.fragment_out(1, Type::uint_t, "out_header");
    info.fragment_out(2, Type::float2_t, "out_normal");
    info.fragment_out(3, Type::float4_t, "out_closure1");
    info.fragment_out(4, Type::float4_t, "out_closure2");

    info.depth_write(DepthWrite::ANY);
    info.storage_buf(0, Qualifier::read, "float", "positions[]");
    info.storage_buf(1, Qualifier::read, "float", "normals[]");
    info.storage_buf(2, Qualifier::read, "float", "uvs[]");
    info.storage_buf(3, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(4, Qualifier::read, "float", "tangents[]");
    info.storage_buf(5, Qualifier::read, "uint", "tri_materials[]");
    info.storage_buf(6, Qualifier::read, "vec4", "material_params[]");
    info.sampler(0, ImageType::Uint2D, "vis_texture");
    info.sampler(1, ImageType::Float2D, "depth_texture");

    info.sampler(2, ImageType::Float2D, "tex_diffuse_0");
    info.sampler(3, ImageType::Float2D, "tex_normal_0");
    info.sampler(4, ImageType::Float2D, "tex_rough_0");

    info.sampler(5, ImageType::Float2D, "tex_diffuse_1");
    info.sampler(6, ImageType::Float2D, "tex_normal_1");
    info.sampler(7, ImageType::Float2D, "tex_rough_1");

    info.sampler(8, ImageType::Float2D, "tex_diffuse_2");
    info.sampler(9, ImageType::Float2D, "tex_normal_2");
    info.sampler(10, ImageType::Float2D, "tex_rough_2");

    info.sampler(11, ImageType::Float2D, "tex_diffuse_3");
    info.sampler(12, ImageType::Float2D, "tex_normal_3");
    info.sampler(13, ImageType::Float2D, "tex_rough_3");

    info.push_constant(Type::float4x4_t, "model_mat");
    info.push_constant(Type::float3x3_t, "normal_mat");
    info.push_constant(Type::int_t, "reverse_z");
    info.push_constant(Type::float_t, "tex_scale_inv");

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

static const char *FORWARD_VERT_SOURCE = R"(
void main()
{
  /* Single Indirect Instanced Draw:
   * - gpu_InstanceIndex is the slot in visible_clusters[] (0 .. instanceCount - 1)
   * - in_vid is the local vertex index within the cluster (0 .. max_cluster_indices - 1) */
  uint slot_idx = uint(gpu_InstanceIndex);
  uint cluster_idx = visible_clusters[slot_idx];
  uvec4 meta = cluster_meta[cluster_idx];
  uint local_vid = in_vid;
  if (local_vid >= meta.y) {
    gl_Position = vec4(2.0, 2.0, 2.0, 0.0);
    v_world_pos = vec3(0.0);
    v_world_normal = vec3(0.0, 0.0, 1.0);
    v_uv = vec2(0.0);
    v_mat = 0u;
    v_cluster_id = 0u;
    return;
  }

  uint v_idx = tri_indices[meta.x + local_vid];
  vec4 v0 = packed_verts[v_idx * 2u + 0u];
  vec4 v1 = packed_verts[v_idx * 2u + 1u];
  vec3 pos = v0.xyz;
  vec3 nor = vec3(v0.w, v1.xy);

  vec4 world_pos = model_mat * vec4(pos, 1.0);
  gl_Position = view_proj * vec4(pos, 1.0);

  mat3 norm_mat = normal_mat;
  vec3 n = norm_mat * nor;
  float len_n = length(n);
  v_world_normal = (len_n > 1e-6) ? (n / len_n) : nor;
  v_world_pos = world_pos.xyz;
  v_cluster_id = cluster_idx;

  if (has_alpha_mask != 0) {
    v_uv = v1.zw;
    uint mat_idx = meta.w;
    if (mat_idx >= 4u) {
      mat_idx = 0u;
    }
    v_mat = mat_idx;
  }
  else {
    v_uv = vec2(0.0);
    v_mat = 0u;
  }
}
)";

static const char *FORWARD_FRAG_SOURCE = R"(
void main()
{
  /* Alpha cutout: leaf / grass cards carry their silhouette in the base
   * color texture alpha channel; reject transparent texel borders so foliage
   * renders as solid, fully-shaped leaves instead of rectangular quads or
   * faint outlines. */
  if (has_alpha_mask != 0) {
    float slot_has_d = (v_mat == 0u) ? has_tex_flags.x :
                       (v_mat == 1u) ? has_tex_flags.y :
                       (v_mat == 2u) ? has_tex_flags.z : has_tex_flags.w;
    if (slot_has_d > 0.5) {
      vec4 tex_col = (v_mat == 0u) ? texture(tex_diffuse_0, v_uv) :
                     (v_mat == 1u) ? texture(tex_diffuse_1, v_uv) :
                     (v_mat == 2u) ? texture(tex_diffuse_2, v_uv) :
                                     texture(tex_diffuse_3, v_uv);
      if (tex_col.a < 0.33) {
        discard;
      }
    }
  }

  if (shading_mode == 1) { /* TESSERA_SHADING_CLUSTERS */
    uint cid = v_cluster_id;
    uint h = cid * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    vec3 col = vec3(float((h >> 16u) & 0xFFu) / 255.0,
                    float((h >> 8u) & 0xFFu) / 255.0,
                    float(h & 0xFFu) / 255.0);
    vec3 N = normalize(v_world_normal);
    if (!gl_FrontFacing) {
      N = -N;
    }
    vec3 L = normalize(camera_pos - v_world_pos);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    out_color = vec4(col * (0.35 + 0.65 * NdotL), 1.0);
    return;
  }
  if (shading_mode == 2) { /* TESSERA_SHADING_NORMALS */
    vec3 N = normalize(v_world_normal);
    if (!gl_FrontFacing) {
      N = -N;
    }
    out_color = vec4(N * 0.5 + 0.5, 1.0);
    return;
  }
  if (shading_mode == 3) { /* TESSERA_SHADING_TANGENTS */
    out_color = vec4(0.5, 0.5, 1.0, 1.0);
    return;
  }
  if (shading_mode == 4) { /* TESSERA_SHADING_UVS */
    out_color = vec4(0.8, 0.8, 0.8, 1.0);
    return;
  }

  /* Default Solid View (Workbench studio / flat shading) */
  vec3 N = normalize(v_world_normal);
  if (!gl_FrontFacing) {
    N = -N;
  }

  if (lighting_type == 0) {
    /* Flat shading */
    out_color = base_color;
  }
  else {
    /* Studio / Headlight shading */
    vec3 L = normalize(camera_pos - v_world_pos);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    vec3 H = normalize(L + vec3(0.2, 0.3, 0.1));
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float spec = pow(NdotH, 32.0) * 0.25;
    vec3 col = base_color.rgb * (0.25 + 0.75 * NdotL) + vec3(spec);
    out_color = vec4(col, base_color.a);
  }
}
)";

/** Create the direct forward rasterization shader for Workbench Solid mode. */
gpu::Shader *createForwardShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("pyGPU_Shader");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;

    info.vertex_source("tessera_forward");
    info.vertex_source_generated = FORWARD_VERT_SOURCE;
    info.fragment_source("tessera_forward");
    info.fragment_source_generated = FORWARD_FRAG_SOURCE;

    info.vertex_in(0, Type::uint_t, "in_vid");
    info.fragment_out(0, Type::float4_t, "out_color");

    info.storage_buf(0, Qualifier::read, "uint", "visible_clusters[]");
    info.storage_buf(1, Qualifier::read, "uvec4", "cluster_meta[]");
    info.storage_buf(2, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(3, Qualifier::read, "vec4", "packed_verts[]");

    info.sampler(0, ImageType::Float2D, "tex_diffuse_0");
    info.sampler(1, ImageType::Float2D, "tex_diffuse_1");
    info.sampler(2, ImageType::Float2D, "tex_diffuse_2");
    info.sampler(3, ImageType::Float2D, "tex_diffuse_3");

    info.push_constant(Type::float4x4_t, "view_proj");
    info.push_constant(Type::float4x4_t, "model_mat");
    info.push_constant(Type::float3x3_t, "normal_mat");
    info.push_constant(Type::float3_t, "camera_pos");
    info.push_constant(Type::float4_t, "base_color");
    info.push_constant(Type::int_t, "shading_mode");
    info.push_constant(Type::int_t, "lighting_type");
    info.push_constant(Type::int_t, "cluster_count");
    info.push_constant(Type::int_t, "debug_clusters");
    info.push_constant(Type::int_t, "has_alpha_mask");
    info.push_constant(Type::float4_t, "has_tex_flags");

    StageInterfaceInfo iface("tessera_forward_iface", "");
    iface.smooth(Type::float3_t, "v_world_pos");
    iface.smooth(Type::float3_t, "v_world_normal");
    iface.smooth(Type::float2_t, "v_uv");
    iface.flat(Type::uint_t, "v_mat");
    iface.flat(Type::uint_t, "v_cluster_id");
    info.vertex_out(iface);

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info_python(raw);
}

}  // namespace

TesseraVisibilityPass::~TesseraVisibilityPass()
{
    terminate();
}

bool TesseraVisibilityPass::init()
{
    if (mDefaultWhiteTex != nullptr) {
        return true;
    }
    if (!gpu_available()) {
        return false;
    }

    /* Create fallback 1x1 textures for unused material texture slots. */
    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    mDefaultWhiteTex = GPU_texture_create_2d("tessera_def_white", 1, 1, 1,
                                             gpu::TextureFormat::UNORM_8_8_8_8,
                                             GPU_TEXTURE_USAGE_SHADER_READ,
                                             white);
    float normal[4] = {0.5f, 0.5f, 1.0f, 1.0f};
    mDefaultNormalTex = GPU_texture_create_2d("tessera_def_normal", 1, 1, 1,
                                              gpu::TextureFormat::UNORM_8_8_8_8,
                                              GPU_TEXTURE_USAGE_SHADER_READ,
                                              normal);
    float rough[4] = {0.5f, 0.5f, 0.5f, 1.0f};
    mDefaultRoughTex = GPU_texture_create_2d("tessera_def_rough", 1, 1, 1,
                                             gpu::TextureFormat::UNORM_8_8_8_8,
                                             GPU_TEXTURE_USAGE_SHADER_READ,
                                             rough);

    return mDefaultWhiteTex != nullptr;
}

bool TesseraVisibilityPass::ensure_vis_resources()
{
    if (!gpu_available()) {
        return false;
    }
    if (mVisShader == nullptr) {
        mVisShader = createVisShader();
    }
    if (mMaterialShader == nullptr) {
        mMaterialShader = createMaterialShader();
    }
    if (mVisShader == nullptr || mMaterialShader == nullptr) {
        return false;
    }
    if (mFullscreenBatch == nullptr) {
        GPUVertFormat fs_format = {};
        uint fs_pos = GPU_vertformat_attr_add(
            &fs_format, "pos", gpu::VertAttrType::SFLOAT_32_32_32);
        mFullscreenVbo = GPU_vertbuf_create_with_format(fs_format);
        GPU_vertbuf_data_alloc(*mFullscreenVbo, 3);
        const float corners[3][3] = {{-1.0f, -1.0f, 0.0f}, {3.0f, -1.0f, 0.0f},
                                     {-1.0f, 3.0f, 0.0f}};
        GPU_vertbuf_attr_fill(mFullscreenVbo, fs_pos, corners);
        mFullscreenBatch = GPU_batch_create_ex(
            GPU_PRIM_TRIS, mFullscreenVbo, nullptr, GPU_BATCH_OWNS_VBO);
    }
    if (mMeshBatch == nullptr) {
        GPUVertFormat idx_format = {};
        GPU_vertformat_attr_add(&idx_format, "vid", gpu::VertAttrType::UINT_32);
        mIndexVbo = GPU_vertbuf_create_with_format(idx_format);
        mMeshBatch = GPU_batch_create_ex(
            GPU_PRIM_TRIS, mIndexVbo, nullptr, GPU_BATCH_OWNS_VBO);
    }
    if (mVisResourcesDirty && mCachedVisData != nullptr) {
        if (!upload_dirty_data()) {
            return false;
        }
        mVisResourcesDirty = false;
    }
    return mPositionsBuf != nullptr && mMeshBatch != nullptr;
}

void TesseraVisibilityPass::terminate()
{
    GPU_FRAMEBUFFER_FREE_SAFE(mFrameBuffer);
    GPU_TEXTURE_FREE_SAFE(mVisTexture);
    GPU_TEXTURE_FREE_SAFE(mDepthTexture);
    GPU_TEXTURE_FREE_SAFE(mDefaultWhiteTex);
    GPU_TEXTURE_FREE_SAFE(mDefaultNormalTex);
    GPU_TEXTURE_FREE_SAFE(mDefaultRoughTex);
    /* The batches own their vertex/index buffers; discarding
     * the batch frees the VBO/IBO — discarding it again would double-free. */
    GPU_BATCH_DISCARD_SAFE(mFullscreenBatch);
    GPU_BATCH_DISCARD_SAFE(mMeshBatch);
    GPU_BATCH_DISCARD_SAFE(mForwardBatch);
    mFullscreenVbo = nullptr;
    mIndexVbo = nullptr;
    mForwardVbo = nullptr;
    mForwardIbo = nullptr;
    GPU_SHADER_FREE_SAFE(mVisShader);
    GPU_SHADER_FREE_SAFE(mMaterialShader);
    GPU_SHADER_FREE_SAFE(mDepthPrepassShader);
    GPU_SHADER_FREE_SAFE(mGbufferShader);
    GPU_SHADER_FREE_SAFE(mForwardShader);
    if (mPackedVertsBuf != nullptr) { GPU_storagebuf_free(mPackedVertsBuf); mPackedVertsBuf = nullptr; }
    if (mPositionsBuf != nullptr) { GPU_storagebuf_free(mPositionsBuf); mPositionsBuf = nullptr; }
    if (mNormalsBuf != nullptr) { GPU_storagebuf_free(mNormalsBuf); mNormalsBuf = nullptr; }
    if (mUvsBuf != nullptr) { GPU_storagebuf_free(mUvsBuf); mUvsBuf = nullptr; }
    if (mTangentsBuf != nullptr) { GPU_storagebuf_free(mTangentsBuf); mTangentsBuf = nullptr; }
    if (mIndicesBuf != nullptr) { GPU_storagebuf_free(mIndicesBuf); mIndicesBuf = nullptr; }
    if (mClusterMetaBuf != nullptr) { GPU_storagebuf_free(mClusterMetaBuf); mClusterMetaBuf = nullptr; }
    if (mClusterMaterialsBuf != nullptr) { GPU_storagebuf_free(mClusterMaterialsBuf); mClusterMaterialsBuf = nullptr; }
    if (mTriMaterialsBuf != nullptr) { GPU_storagebuf_free(mTriMaterialsBuf); mTriMaterialsBuf = nullptr; }
    if (mMaterialParamsBuf != nullptr) { GPU_storagebuf_free(mMaterialParamsBuf); mMaterialParamsBuf = nullptr; }
    mCachedVisData = nullptr;
    mVisResourcesDirty = false;
    mMaxClusterIndices = 384;
    mVertexData = TesseraGpuVertexData();
    mVertexCount = 0;
    mIndexCount = 0;
    mClusterCount = 0;
    mWidth = 0;
    mHeight = 0;
}

bool TesseraVisibilityPass::ensure_targets(int width, int height)
{
    if (mVisTexture != nullptr && mWidth == width && mHeight == height) {
        return true;
    }
    GPU_TEXTURE_FREE_SAFE(mVisTexture);
    GPU_TEXTURE_FREE_SAFE(mDepthTexture);
    GPU_FRAMEBUFFER_FREE_SAFE(mFrameBuffer);
    mWidth = width;
    mHeight = height;

    mVisTexture = GPU_texture_create_2d("tessera_vis_buffer",
                                        width,
                                        height,
                                        1,
                                        gpu::TextureFormat::UINT_32_32_32_32,
                                        GPU_TEXTURE_USAGE_ATTACHMENT | GPU_TEXTURE_USAGE_SHADER_READ,
                                        nullptr);
    mDepthTexture = GPU_texture_create_2d("tessera_vis_depth",
                                          width,
                                          height,
                                          1,
                                          gpu::TextureFormat::SFLOAT_32_DEPTH,
                                          GPU_TEXTURE_USAGE_ATTACHMENT | GPU_TEXTURE_USAGE_SHADER_READ,
                                          nullptr);
    if (mVisTexture == nullptr || mDepthTexture == nullptr) {
        return false;
    }
    mFrameBuffer = GPU_framebuffer_create("tessera_vis_fb");
    GPU_framebuffer_ensure_config(&mFrameBuffer,
                                  {GPU_ATTACHMENT_TEXTURE(mDepthTexture),
                                   GPU_ATTACHMENT_TEXTURE(mVisTexture)});
    return mFrameBuffer != nullptr;
}

bool TesseraVisibilityPass::upload_dirty_data()
{
    /* Lazily build the visibility-buffer resources (positions/normals/uvs/tangents SSBOs, tri
     * materials, vertex-pulling identity VBO). Only needed by render(), render_depth_prepass()
     * and render_gbuffer(); the Workbench forward path never pays this memory/upload cost. */
    if (mCachedVisData == nullptr) {
        return false;
    }
    const TesseraData &data = *mCachedVisData;
    flattenVertexDataForGpu(data, mVertexData);
    const size_t index_count = mVertexData.indices.size();
    if (index_count == 0) {
        return false;
    }
    if (mPositionsBuf != nullptr) { GPU_storagebuf_free(mPositionsBuf); mPositionsBuf = nullptr; }
    if (mNormalsBuf != nullptr) { GPU_storagebuf_free(mNormalsBuf); mNormalsBuf = nullptr; }
    if (mUvsBuf != nullptr) { GPU_storagebuf_free(mUvsBuf); mUvsBuf = nullptr; }
    if (mTangentsBuf != nullptr) { GPU_storagebuf_free(mTangentsBuf); mTangentsBuf = nullptr; }
    if (mTriMaterialsBuf != nullptr) { GPU_storagebuf_free(mTriMaterialsBuf); mTriMaterialsBuf = nullptr; }
    mPositionsBuf = GPU_storagebuf_create_ex(
        std::max<size_t>(16, mVertexData.positions.size() * sizeof(float)),
        nullptr,
        GPU_USAGE_STATIC,
        "tessera_positions");
    mNormalsBuf = GPU_storagebuf_create_ex(
        std::max<size_t>(16, mVertexData.normals.size() * sizeof(float)),
        nullptr,
        GPU_USAGE_STATIC,
        "tessera_normals");
    mUvsBuf = GPU_storagebuf_create_ex(std::max<size_t>(16, mVertexData.uvs.size() * sizeof(float)),
                                       nullptr,
                                       GPU_USAGE_STATIC,
                                       "tessera_uvs");
    mTangentsBuf = GPU_storagebuf_create_ex(
        std::max<size_t>(16, mVertexData.tangents.size() * sizeof(float)),
        nullptr,
        GPU_USAGE_STATIC,
        "tessera_tangents");
    if (mPositionsBuf == nullptr || mNormalsBuf == nullptr || mUvsBuf == nullptr ||
        mTangentsBuf == nullptr || mIndicesBuf == nullptr)
    {
        return false;
    }
    if (!mVertexData.positions.empty()) {
        GPU_storagebuf_update(mPositionsBuf, mVertexData.positions.data());
    }
    if (!mVertexData.normals.empty()) {
        GPU_storagebuf_update(mNormalsBuf, mVertexData.normals.data());
    }
    if (!mVertexData.uvs.empty()) {
        GPU_storagebuf_update(mUvsBuf, mVertexData.uvs.data());
    }
    if (!mVertexData.tangents.empty()) {
        GPU_storagebuf_update(mTangentsBuf, mVertexData.tangents.data());
    }

    const size_t num_tris = index_count / 3;
    std::vector<uint32_t> tri_materials(std::max(size_t(1), num_tris), 0);
    for (const auto &nc : data.clusters) {
        uint32_t start_tri = nc.firstIndex / 3;
        uint32_t count_tri = nc.indexCount / 3;
        for (uint32_t k = 0; k < count_tri && (start_tri + k) < num_tris; ++k) {
            tri_materials[start_tri + k] = uint32_t(nc.materialIndex);
        }
    }
    mTriMaterialsBuf = GPU_storagebuf_create_ex(tri_materials.size() * sizeof(uint32_t),
                                                tri_materials.data(),
                                                GPU_USAGE_STATIC,
                                                "tessera_tri_materials");

    /* Grow the vertex-pulling batch (one dummy attribute per corner). */
    GPU_vertbuf_data_alloc(*mIndexVbo, uint(index_count));
    std::vector<uint32_t> identity(index_count);
    for (size_t i = 0; i < index_count; ++i) {
        identity[i] = uint32_t(i);
    }
    GPU_vertbuf_attr_fill(mIndexVbo, 0, identity.data());

    /* Release the CPU-side staging copy; the data now lives in VRAM. */
    mVertexData = TesseraGpuVertexData();
    return true;
}

void TesseraVisibilityPass::render(gpu::StorageBuf *command_buf,
                                  size_t draw_count,
                                  const float4x4 &viewProj,
                                  const float3 &cam_pos,
                                  int viewport_width,
                                  int viewport_height,
                                  int shading_mode,
                                  const float4x4 &object_model,
                                  bool reverse_z,
                                  const std::vector<uint32_t> *material_counts,
                                  uint32_t max_commands_per_material,
                                  gpu::StorageBuf *counter_buf)
{
    if (command_buf == nullptr || draw_count == 0 || !ensure_vis_resources()) {
        return;
    }
    if (command_buf == nullptr || draw_count == 0 || mVisShader == nullptr ||
        mMaterialShader == nullptr || mPositionsBuf == nullptr ||
        mMeshBatch == nullptr)
    {
        return;
    }
    if (viewport_width <= 0 || viewport_height <= 0) {
        return;
    }
    if (!ensure_targets(viewport_width, viewport_height)) {
        return;
    }

    gpu::FrameBuffer *previous_fb = GPU_framebuffer_active_get();

    /* Encode "no geometry" as 0u (and 0xFFFFFFFF). */
    const uint32_t clear_value[4] = {0u, 0u, 0u, 0u};
    GPU_texture_clear(
        mVisTexture, GPU_DATA_UINT, static_cast<const void *>(clear_value));

    /* ---------------- Pass 1: visibility ---------------- */
    GPU_framebuffer_bind(mFrameBuffer);
    GPU_framebuffer_clear_depth(mFrameBuffer, reverse_z ? 0.0f : 1.0f);
    GPU_clip_control_unit_range(reverse_z);
    GPU_depth_test(reverse_z ? GPU_DEPTH_GREATER_EQUAL : GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_face_culling(GPU_CULL_NONE);
    GPU_blend(GPU_BLEND_NONE);
    GPU_color_mask(true, true, true, true);

    GPU_shader_bind(mVisShader);
    GPU_storagebuf_bind(mIndicesBuf, 0);
    GPU_storagebuf_bind(mPositionsBuf, 1);
    GPU_shader_uniform_mat4(mVisShader, "view_proj", viewProj.ptr());
    GPU_shader_uniform_1i(
        mVisShader,
        "debug_clusters",
        shading_mode == TESSERA_SHADING_CLUSTERS ? 1 : 0);
    GPU_shader_uniform_1i(mVisShader, "cluster_count", int(mClusterCount));
    GPU_shader_uniform_1i(mVisShader, "reverse_z", reverse_z ? 1 : 0);

    const intptr_t cmd_stride = sizeof(uint32_t) * 4;
    if (counter_buf != nullptr && max_commands_per_material > 0) {
        GPU_storagebuf_sync_as_indirect_buffer(counter_buf);
        const size_t num_mats = (material_counts != nullptr && !material_counts->empty()) ?
                                    material_counts->size() :
                                    1;
        const int max_cnt = int(std::min<size_t>(mClusterCount, max_commands_per_material));
        for (size_t mat = 0; mat < num_mats; ++mat) {
            intptr_t ind_offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            intptr_t cnt_offset = intptr_t(mat) * sizeof(uint32_t);
            GPU_batch_multi_draw_indirect_count(
                mMeshBatch, command_buf, counter_buf, max_cnt, ind_offset, cnt_offset, cmd_stride);
        }
    }
    else if (material_counts != nullptr && !material_counts->empty() && max_commands_per_material > 0) {
        for (size_t mat = 0; mat < material_counts->size(); ++mat) {
            uint32_t count = (*material_counts)[mat];
            if (count == 0) continue;
            intptr_t offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            GPU_batch_multi_draw_indirect(
                mMeshBatch, command_buf, int(count), offset, cmd_stride);
        }
    }
    else {
        GPU_batch_multi_draw_indirect(
            mMeshBatch, command_buf, int(draw_count), 0, cmd_stride);
    }

    GPU_color_mask(true, true, true, true);
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

    /* ---------------- Pass 2: material ---------------- */
    GPU_framebuffer_bind(previous_fb != nullptr ? previous_fb : mFrameBuffer);
    GPU_clip_control_unit_range(reverse_z);
    GPU_depth_test(reverse_z ? GPU_DEPTH_GREATER_EQUAL : GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_blend(GPU_BLEND_NONE);

    GPU_batch_set_shader(mFullscreenBatch, mMaterialShader);
    GPU_storagebuf_bind(mPositionsBuf, 0);
    GPU_storagebuf_bind(mNormalsBuf, 1);
    GPU_storagebuf_bind(mUvsBuf, 2);
    GPU_storagebuf_bind(mIndicesBuf, 3);
    if (mTangentsBuf != nullptr) {
        GPU_storagebuf_bind(mTangentsBuf, 4);
    }
    if (mTriMaterialsBuf != nullptr) {
        GPU_storagebuf_bind(mTriMaterialsBuf, 5);
    }

    /* Upload material parameters: 4 vec4 per slot (base_color, props, rough_params, extra) */
    float mat_params[4 * 16] = {};
    for (int s = 0; s < 4; ++s) {
        if (s < int(mMaterials.size())) {
            const auto &m = mMaterials[s];
            mat_params[s * 16 + 0] = m.base_color.x;
            mat_params[s * 16 + 1] = m.base_color.y;
            mat_params[s * 16 + 2] = m.base_color.z;
            mat_params[s * 16 + 3] = m.metallic_value;
            mat_params[s * 16 + 4] = (m.has_diffuse && m.diffuse != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 5] = (m.has_normal && m.normal != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 6] = (m.has_roughness && m.roughness != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 7] = m.specular_value;
            mat_params[s * 16 + 8] = m.roughness_value;
            mat_params[s * 16 + 9] = m.roughness_gamma;
            mat_params[s * 16 + 10] = m.roughness_scale;
            mat_params[s * 16 + 11] = m.roughness_offset;
            mat_params[s * 16 + 12] = m.diffuse_is_srgb ? 1.0f : 0.0f;
            mat_params[s * 16 + 13] = m.has_specular_tint ? 1.0f : 0.0f;
            mat_params[s * 16 + 14] = 0.0f;
            mat_params[s * 16 + 15] = 0.0f;
        } else {
            mat_params[s * 16 + 0] = 0.8f;
            mat_params[s * 16 + 1] = 0.8f;
            mat_params[s * 16 + 2] = 0.8f;
            mat_params[s * 16 + 3] = 0.0f;
            mat_params[s * 16 + 4] = 0.0f;
            mat_params[s * 16 + 5] = 0.0f;
            mat_params[s * 16 + 6] = 0.0f;
            mat_params[s * 16 + 7] = 0.5f;
            mat_params[s * 16 + 8] = 0.5f;
            mat_params[s * 16 + 9] = 1.0f;
            mat_params[s * 16 + 10] = 1.0f;
            mat_params[s * 16 + 11] = 0.0f;
            mat_params[s * 16 + 12] = 0.0f;
            mat_params[s * 16 + 13] = 0.0f;
            mat_params[s * 16 + 14] = 0.0f;
            mat_params[s * 16 + 15] = 0.0f;
        }
    }
    if (mMaterialParamsBuf == nullptr) {
        mMaterialParamsBuf = GPU_storagebuf_create_ex(
            sizeof(mat_params), mat_params, GPU_USAGE_DYNAMIC, "tessera_mat_params");
    } else {
        GPU_storagebuf_update(mMaterialParamsBuf, mat_params);
    }
    GPU_storagebuf_bind(mMaterialParamsBuf, 6);

    GPU_texture_bind(mVisTexture, 0);
    GPU_texture_bind(mDepthTexture, 1);

    for (int s = 0; s < 4; ++s) {
        gpu::Texture *td = mDefaultWhiteTex;
        gpu::Texture *tn = mDefaultNormalTex;
        gpu::Texture *tr = mDefaultRoughTex;
        if (s < int(mMaterials.size())) {
            const auto &m = mMaterials[s];
            if (m.has_diffuse && m.diffuse) td = m.diffuse;
            if (m.has_normal && m.normal) tn = m.normal;
            if (m.has_roughness && m.roughness) tr = m.roughness;
        }
        GPU_texture_bind(td, 2 + s * 3 + 0);
        GPU_texture_bind(tn, 2 + s * 3 + 1);
        GPU_texture_bind(tr, 2 + s * 3 + 2);
    }

    GPU_shader_uniform_mat4(mMaterialShader, "model_mat", object_model.ptr());
    float3x3 mat_norm = compute_normal_matrix(object_model);
    GPU_shader_uniform_mat3(mMaterialShader, "normal_mat", reinterpret_cast<const float(*)[3]>(mat_norm.ptr()));
    GPU_shader_uniform_3f(
        mMaterialShader, "camera_pos", cam_pos.x, cam_pos.y, cam_pos.z);
    GPU_shader_uniform_1i(mMaterialShader, "shading_mode", shading_mode);
    GPU_shader_uniform_1i(mMaterialShader, "reverse_z", reverse_z ? 1 : 0);
    GPU_shader_uniform_1f(mMaterialShader, "tex_scale_inv", 1.0f / mTextureScale);

    bool use_scissor = false;
    int sc_x = 0, sc_y = 0, sc_w = viewport_width, sc_h = viewport_height;
    float bmin[3] = {mBoundsMin[0], mBoundsMin[1], mBoundsMin[2]};
    float bmax[3] = {mBoundsMax[0], mBoundsMax[1], mBoundsMax[2]};
    if (bmin[0] < bmax[0] && bmin[1] < bmax[1] && bmin[2] < bmax[2]) {
        float min_x = 1e9f, min_y = 1e9f;
        float max_x = -1e9f, max_y = -1e9f;
        bool all_front = true;
        float4x4 mvp = viewProj;
        for (int i = 0; i < 8; ++i) {
            float4 corner(
                (i & 1) ? bmax[0] : bmin[0],
                (i & 2) ? bmax[1] : bmin[1],
                (i & 4) ? bmax[2] : bmin[2],
                1.0f
            );
            float4 clip = mvp * corner;
            if (clip.w <= 0.001f) {
                all_front = false;
                break;
            }
            float inv_w = 1.0f / clip.w;
            float sx = (clip.x * inv_w * 0.5f + 0.5f) * float(viewport_width);
            float sy = (clip.y * inv_w * 0.5f + 0.5f) * float(viewport_height);
            min_x = std::min(min_x, sx);
            min_y = std::min(min_y, sy);
            max_x = std::max(max_x, sx);
            max_y = std::max(max_y, sy);
        }
        if (all_front && max_x > min_x && max_y > min_y) {
            int px0 = std::clamp(int(std::floor(min_x)) - 2, 0, viewport_width);
            int py0 = std::clamp(int(std::floor(min_y)) - 2, 0, viewport_height);
            int px1 = std::clamp(int(std::ceil(max_x)) + 2, 0, viewport_width);
            int py1 = std::clamp(int(std::ceil(max_y)) + 2, 0, viewport_height);
            if (px1 > px0 && py1 > py0) {
                sc_x = px0;
                sc_y = py0;
                sc_w = px1 - px0;
                sc_h = py1 - py0;
                use_scissor = true;
            }
        }
    }

    if (use_scissor) {
        GPU_scissor_test(true);
        GPU_scissor(sc_x, sc_y, sc_w, sc_h);
    }
    GPU_batch_draw(mFullscreenBatch);
    if (use_scissor) {
        GPU_scissor_test(false);
    }
}

void TesseraVisibilityPass::render_depth_prepass(gpu::StorageBuf *command_buf,
                                                size_t draw_count,
                                                const float4x4 &viewProj,
                                                int viewport_width,
                                                int viewport_height,
                                                bool reverse_z,
                                                const std::vector<uint32_t> *material_counts,
                                                uint32_t max_commands_per_material,
                                                gpu::StorageBuf *counter_buf)
{
    if (command_buf == nullptr || draw_count == 0 || !ensure_vis_resources()) {
        return;
    }
    if (command_buf == nullptr || draw_count == 0 || mMeshBatch == nullptr ||
        mPositionsBuf == nullptr || mIndicesBuf == nullptr)
    {
        return;
    }
    if (viewport_width <= 0 || viewport_height <= 0) {
        return;
    }
    if (mDepthPrepassShader == nullptr) {
        mDepthPrepassShader = createDepthPrepassShader();
        if (mDepthPrepassShader == nullptr) {
            return;
        }
    }

    GPU_clip_control_unit_range(reverse_z);
    GPU_depth_test(reverse_z ? GPU_DEPTH_GREATER_EQUAL : GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_color_mask(false, false, false, false);
    /* Face culling stays disabled: two-sided foliage cards must rasterize
     * their back faces, and occlusion is resolved by depth. */
    GPU_face_culling(GPU_CULL_NONE);
    GPU_blend(GPU_BLEND_NONE);

    GPU_shader_bind(mDepthPrepassShader);
    GPU_storagebuf_bind(mIndicesBuf, 0);
    GPU_storagebuf_bind(mPositionsBuf, 1);
    GPU_storagebuf_bind(mUvsBuf, 2);
    GPU_storagebuf_bind(mTriMaterialsBuf, 3);

    /* Alpha cutout state (kept in sync with the material/forward passes so
     * the depth buffer matches the shaded silhouette). */
    bool depth_alpha_active = false;
    float4 depth_has_tex(0.0f);
    for (int s = 0; s < 4; ++s) {
        const bool slot_has_diffuse = (s < int(mMaterials.size())) &&
                                      mMaterials[s].has_diffuse &&
                                      (mMaterials[s].diffuse != nullptr);
        gpu::Texture *diff = slot_has_diffuse ?
                                 mMaterials[s].diffuse :
                                 ((mDefaultWhiteTex != nullptr) ? mDefaultWhiteTex : nullptr);
        if (diff != nullptr) {
            GPU_texture_bind(diff, s);
        }
        if (slot_has_diffuse) {
            depth_has_tex[s] = 1.0f;
            depth_alpha_active = true;
        }
    }
    depth_alpha_active = depth_alpha_active && mAlphaMask;

    GPU_shader_uniform_mat4(mDepthPrepassShader, "view_proj", viewProj.ptr());
    GPU_shader_uniform_1i(mDepthPrepassShader, "reverse_z", reverse_z ? 1 : 0);
    GPU_shader_uniform_1i(mDepthPrepassShader, "has_alpha_mask", depth_alpha_active ? 1 : 0);
    GPU_shader_uniform_4f(mDepthPrepassShader, "has_tex_flags", depth_has_tex.x, depth_has_tex.y, depth_has_tex.z, depth_has_tex.w);

    const intptr_t cmd_stride = sizeof(uint32_t) * 4;
    if (counter_buf != nullptr && max_commands_per_material > 0) {
        GPU_storagebuf_sync_as_indirect_buffer(counter_buf);
        const size_t num_mats = (material_counts != nullptr && !material_counts->empty()) ?
                                    material_counts->size() :
                                    1;
        const int max_cnt = int(std::min<size_t>(mClusterCount, max_commands_per_material));
        for (size_t mat = 0; mat < num_mats; ++mat) {
            intptr_t ind_offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            intptr_t cnt_offset = intptr_t(mat) * sizeof(uint32_t);
            GPU_batch_multi_draw_indirect_count(
                mMeshBatch, command_buf, counter_buf, max_cnt, ind_offset, cnt_offset, cmd_stride);
        }
    }
    else if (material_counts != nullptr && !material_counts->empty() && max_commands_per_material > 0) {
        for (size_t mat = 0; mat < material_counts->size(); ++mat) {
            uint32_t count = (*material_counts)[mat];
            if (count == 0) continue;
            intptr_t offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            GPU_batch_multi_draw_indirect(
                mMeshBatch, command_buf, int(count), offset, cmd_stride);
        }
    }
    else {
        GPU_batch_multi_draw_indirect(
            mMeshBatch, command_buf, int(draw_count), 0, cmd_stride);
    }

    GPU_color_mask(true, true, true, true);
}

void TesseraVisibilityPass::render_gbuffer(gpu::StorageBuf *command_buf,
                                          size_t draw_count,
                                          const float4x4 &viewProj,
                                          const float3 &cam_pos,
                                          int viewport_width,
                                          int viewport_height,
                                          const float4x4 &object_model,
                                          bool reverse_z,
                                          const std::vector<uint32_t> *material_counts,
                                          uint32_t max_commands_per_material,
                                          gpu::StorageBuf *counter_buf)
{
    if (command_buf == nullptr || draw_count == 0 || !ensure_vis_resources()) {
        return;
    }
    if (command_buf == nullptr || draw_count == 0 || mVisShader == nullptr ||
        mPositionsBuf == nullptr || mMeshBatch == nullptr)
    {
        return;
    }
    if (viewport_width <= 0 || viewport_height <= 0) {
        return;
    }
    if (!ensure_targets(viewport_width, viewport_height)) {
        return;
    }
    if (mGbufferShader == nullptr) {
        mGbufferShader = createGbufferShader();
        if (mGbufferShader == nullptr) {
            return;
        }
    }

    (void)cam_pos;
    gpu::FrameBuffer *target_gbuffer_fb = GPU_framebuffer_active_get();

    /* Encode "no geometry" as 0u (and 0xFFFFFFFF). */
    const uint32_t clear_value[4] = {0u, 0u, 0u, 0u};
    GPU_texture_clear(mVisTexture, GPU_DATA_UINT, static_cast<const void *>(clear_value));

    /* ---------------- Pass 1: visibility ---------------- */
    GPU_framebuffer_bind(mFrameBuffer);
    GPU_framebuffer_clear_depth(mFrameBuffer, reverse_z ? 0.0f : 1.0f);
    GPU_clip_control_unit_range(reverse_z);
    GPU_depth_test(reverse_z ? GPU_DEPTH_GREATER_EQUAL : GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_face_culling(GPU_CULL_NONE);
    GPU_blend(GPU_BLEND_NONE);
    GPU_color_mask(true, true, true, true);

    GPU_shader_bind(mVisShader);
    GPU_storagebuf_bind(mIndicesBuf, 0);
    GPU_storagebuf_bind(mPositionsBuf, 1);
    GPU_shader_uniform_mat4(mVisShader, "view_proj", viewProj.ptr());
    GPU_shader_uniform_1i(mVisShader, "debug_clusters", 0);
    GPU_shader_uniform_1i(mVisShader, "cluster_count", int(mClusterCount));
    GPU_shader_uniform_1i(mVisShader, "reverse_z", reverse_z ? 1 : 0);

    const intptr_t cmd_stride = sizeof(uint32_t) * 4;
    if (counter_buf != nullptr && max_commands_per_material > 0) {
        GPU_storagebuf_sync_as_indirect_buffer(counter_buf);
        const size_t num_mats = (material_counts != nullptr && !material_counts->empty()) ?
                                    material_counts->size() :
                                    1;
        const int max_cnt = int(std::min<size_t>(mClusterCount, max_commands_per_material));
        for (size_t mat = 0; mat < num_mats; ++mat) {
            intptr_t ind_offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            intptr_t cnt_offset = intptr_t(mat) * sizeof(uint32_t);
            GPU_batch_multi_draw_indirect_count(
                mMeshBatch, command_buf, counter_buf, max_cnt, ind_offset, cnt_offset, cmd_stride);
        }
    }
    else if (material_counts != nullptr && !material_counts->empty() && max_commands_per_material > 0) {
        for (size_t mat = 0; mat < material_counts->size(); ++mat) {
            uint32_t count = (*material_counts)[mat];
            if (count == 0) continue;
            intptr_t offset = intptr_t(mat) * max_commands_per_material * cmd_stride;
            GPU_batch_multi_draw_indirect(
                mMeshBatch, command_buf, int(count), offset, cmd_stride);
        }
    }
    else {
        GPU_batch_multi_draw_indirect(
            mMeshBatch, command_buf, int(draw_count), 0, cmd_stride);
    }

    GPU_color_mask(true, true, true, true);
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

    /* ---------------- Pass 2: G-Buffer output ---------------- */
    GPU_framebuffer_bind(target_gbuffer_fb);
    GPU_clip_control_unit_range(reverse_z);
    GPU_depth_test(reverse_z ? GPU_DEPTH_GREATER_EQUAL : GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_blend(GPU_BLEND_NONE);

    GPU_batch_set_shader(mFullscreenBatch, mGbufferShader);
    GPU_storagebuf_bind(mPositionsBuf, 0);
    GPU_storagebuf_bind(mNormalsBuf, 1);
    GPU_storagebuf_bind(mUvsBuf, 2);
    GPU_storagebuf_bind(mIndicesBuf, 3);
    if (mTangentsBuf != nullptr) {
        GPU_storagebuf_bind(mTangentsBuf, 4);
    }
    if (mTriMaterialsBuf != nullptr) {
        GPU_storagebuf_bind(mTriMaterialsBuf, 5);
    }

    /* Upload material parameters: 4 vec4 per slot (base_color, props, rough_params, extra) */
    float mat_params[4 * 16] = {};
    for (int s = 0; s < 4; ++s) {
        if (s < int(mMaterials.size())) {
            const auto &m = mMaterials[s];
            mat_params[s * 16 + 0] = m.base_color.x;
            mat_params[s * 16 + 1] = m.base_color.y;
            mat_params[s * 16 + 2] = m.base_color.z;
            mat_params[s * 16 + 3] = m.metallic_value;
            mat_params[s * 16 + 4] = (m.has_diffuse && m.diffuse != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 5] = (m.has_normal && m.normal != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 6] = (m.has_roughness && m.roughness != nullptr) ? 1.0f : 0.0f;
            mat_params[s * 16 + 7] = m.specular_value;
            mat_params[s * 16 + 8] = m.roughness_value;
            mat_params[s * 16 + 9] = m.roughness_gamma;
            mat_params[s * 16 + 10] = m.roughness_scale;
            mat_params[s * 16 + 11] = m.roughness_offset;
            mat_params[s * 16 + 12] = m.diffuse_is_srgb ? 1.0f : 0.0f;
            mat_params[s * 16 + 13] = m.has_specular_tint ? 1.0f : 0.0f;
            mat_params[s * 16 + 14] = 0.0f;
            mat_params[s * 16 + 15] = 0.0f;
        } else {
            mat_params[s * 16 + 0] = 0.8f;
            mat_params[s * 16 + 1] = 0.8f;
            mat_params[s * 16 + 2] = 0.8f;
            mat_params[s * 16 + 3] = 0.0f;
            mat_params[s * 16 + 4] = 0.0f;
            mat_params[s * 16 + 5] = 0.0f;
            mat_params[s * 16 + 6] = 0.0f;
            mat_params[s * 16 + 7] = 0.5f;
            mat_params[s * 16 + 8] = 0.5f;
            mat_params[s * 16 + 9] = 1.0f;
            mat_params[s * 16 + 10] = 1.0f;
            mat_params[s * 16 + 11] = 0.0f;
            mat_params[s * 16 + 12] = 0.0f;
            mat_params[s * 16 + 13] = 0.0f;
            mat_params[s * 16 + 14] = 0.0f;
            mat_params[s * 16 + 15] = 0.0f;
        }
    }
    if (mMaterialParamsBuf == nullptr) {
        mMaterialParamsBuf = GPU_storagebuf_create_ex(
            sizeof(mat_params), mat_params, GPU_USAGE_DYNAMIC, "tessera_mat_params");
    } else {
        GPU_storagebuf_update(mMaterialParamsBuf, mat_params);
    }
    GPU_storagebuf_bind(mMaterialParamsBuf, 6);

    GPU_texture_bind(mVisTexture, 0);
    GPU_texture_bind(mDepthTexture, 1);

    for (int s = 0; s < 4; ++s) {
        gpu::Texture *td = mDefaultWhiteTex;
        gpu::Texture *tn = mDefaultNormalTex;
        gpu::Texture *tr = mDefaultRoughTex;
        if (s < int(mMaterials.size())) {
            const auto &m = mMaterials[s];
            if (m.has_diffuse && m.diffuse) td = m.diffuse;
            if (m.has_normal && m.normal) tn = m.normal;
            if (m.has_roughness && m.roughness) tr = m.roughness;
        }
        GPU_texture_bind(td, 2 + s * 3 + 0);
        GPU_texture_bind(tn, 2 + s * 3 + 1);
        GPU_texture_bind(tr, 2 + s * 3 + 2);
    }

    GPU_shader_uniform_mat4(mGbufferShader, "model_mat", object_model.ptr());
    float3x3 gbuf_norm = compute_normal_matrix(object_model);
    GPU_shader_uniform_mat3(mGbufferShader, "normal_mat", reinterpret_cast<const float(*)[3]>(gbuf_norm.ptr()));
    GPU_shader_uniform_1i(mGbufferShader, "reverse_z", reverse_z ? 1 : 0);
    GPU_shader_uniform_1f(mGbufferShader, "tex_scale_inv", 1.0f / mTextureScale);

    bool use_scissor = false;
    int sc_x = 0, sc_y = 0, sc_w = viewport_width, sc_h = viewport_height;
    float bmin[3] = {mBoundsMin[0], mBoundsMin[1], mBoundsMin[2]};
    float bmax[3] = {mBoundsMax[0], mBoundsMax[1], mBoundsMax[2]};
    if (bmin[0] < bmax[0] && bmin[1] < bmax[1] && bmin[2] < bmax[2]) {
        float min_x = 1e9f, min_y = 1e9f;
        float max_x = -1e9f, max_y = -1e9f;
        bool all_front = true;
        float4x4 mvp = viewProj;
        for (int i = 0; i < 8; ++i) {
            float4 corner(
                (i & 1) ? bmax[0] : bmin[0],
                (i & 2) ? bmax[1] : bmin[1],
                (i & 4) ? bmax[2] : bmin[2],
                1.0f
            );
            float4 clip = mvp * corner;
            if (clip.w <= 0.001f) {
                all_front = false;
                break;
            }
            float inv_w = 1.0f / clip.w;
            float sx = (clip.x * inv_w * 0.5f + 0.5f) * float(viewport_width);
            float sy = (clip.y * inv_w * 0.5f + 0.5f) * float(viewport_height);
            min_x = std::min(min_x, sx);
            min_y = std::min(min_y, sy);
            max_x = std::max(max_x, sx);
            max_y = std::max(max_y, sy);
        }
        if (all_front && max_x > min_x && max_y > min_y) {
            int px0 = std::clamp(int(std::floor(min_x)) - 2, 0, viewport_width);
            int py0 = std::clamp(int(std::floor(min_y)) - 2, 0, viewport_height);
            int px1 = std::clamp(int(std::ceil(max_x)) + 2, 0, viewport_width);
            int py1 = std::clamp(int(std::ceil(max_y)) + 2, 0, viewport_height);
            if (px1 > px0 && py1 > py0) {
                sc_x = px0;
                sc_y = py0;
                sc_w = px1 - px0;
                sc_h = py1 - py0;
                use_scissor = true;
            }
        }
    }

    if (use_scissor) {
        GPU_scissor_test(true);
        GPU_scissor(sc_x, sc_y, sc_w, sc_h);
    }
    GPU_batch_draw(mFullscreenBatch);
    if (use_scissor) {
        GPU_scissor_test(false);
    }
}


void TesseraVisibilityPass::set_data(const TesseraData &data)
{
    if (!gpu_available()) {
        return;
    }
    const size_t vertex_count = data.positions.size();
    const size_t index_count = data.indices.size();
    const size_t cluster_count = data.clusters.size();
    if (vertex_count == 0 || index_count == 0 || cluster_count == 0) {
        return;
    }
    mCachedVisData = &data;
    if (vertex_count == mVertexCount && index_count == mIndexCount &&
        cluster_count == mClusterCount && mForwardBatch != nullptr)
    {
        return;
    }
    mBoundsMin[0] = data.header.boundsMin[0];
    mBoundsMin[1] = data.header.boundsMin[1];
    mBoundsMin[2] = data.header.boundsMin[2];
    mBoundsMax[0] = data.header.boundsMax[0];
    mBoundsMax[1] = data.header.boundsMax[1];
    mBoundsMax[2] = data.header.boundsMax[2];

    mVertexCount = vertex_count;
    mIndexCount = index_count;
    mClusterCount = cluster_count;
    mVisResourcesDirty = true;

    /* ---- Persistent, upload-once forward resources (all GPU_USAGE_STATIC) ---- */

    /* Packed vertices: 2 x vec4 per vertex = (px, py, pz, nx), (ny, nz, u, v). */
    const bool has_nor = data.normals.size() >= vertex_count;
    const bool has_uv = data.uvs.size() >= vertex_count;
    std::vector<float> packed(vertex_count * 8);
    for (size_t i = 0; i < vertex_count; ++i) {
        const float3 &p = data.positions[i];
        const float3 n = has_nor ? data.normals[i] : float3(0.0f, 0.0f, 1.0f);
        const float2 uv = has_uv ? data.uvs[i] : float2(0.0f);
        float *dst = &packed[i * 8];
        dst[0] = p.x;
        dst[1] = p.y;
        dst[2] = p.z;
        dst[3] = n.x;
        dst[4] = n.y;
        dst[5] = n.z;
        dst[6] = uv.x;
        dst[7] = uv.y;
    }
    if (mPackedVertsBuf != nullptr) {
        GPU_storagebuf_free(mPackedVertsBuf);
    }
    mPackedVertsBuf = GPU_storagebuf_create_ex(
        packed.size() * sizeof(float), packed.data(), GPU_USAGE_STATIC, "tessera_packed_verts");

    if (mIndicesBuf != nullptr) {
        GPU_storagebuf_free(mIndicesBuf);
    }
    mIndicesBuf = GPU_storagebuf_create_ex(index_count * sizeof(uint32_t),
                                           data.indices.data(),
                                           GPU_USAGE_STATIC,
                                           "tessera_tri_indices");

    /* Cluster metadata: uvec4(firstIndex, indexCount, level, materialIndex). */
    uint32_t max_cluster_idx = 384;
    std::vector<uint32_t> meta(cluster_count * 4);
    std::vector<uint32_t> cluster_materials(cluster_count, 0);
    for (size_t c = 0; c < cluster_count; ++c) {
        const TesseraCluster &cl = data.clusters[c];
        uint32_t count = cl.indexCount;
        if (size_t(cl.firstIndex) + count > index_count) {
            count = (cl.firstIndex < index_count) ? uint32_t(index_count - cl.firstIndex) : 0u;
            count -= count % 3;
        }
        meta[c * 4 + 0] = cl.firstIndex;
        meta[c * 4 + 1] = count;
        meta[c * 4 + 2] = cl.level;
        meta[c * 4 + 3] = uint32_t(cl.materialIndex);
        cluster_materials[c] = uint32_t(cl.materialIndex);
        max_cluster_idx = std::max(max_cluster_idx, cl.indexCount);
    }
    mMaxClusterIndices = ((max_cluster_idx + 2) / 3) * 3;
    if (mClusterMetaBuf != nullptr) {
        GPU_storagebuf_free(mClusterMetaBuf);
    }
    mClusterMetaBuf = GPU_storagebuf_create_ex(
        meta.size() * sizeof(uint32_t), meta.data(), GPU_USAGE_STATIC, "tessera_vis_cluster_meta");
    if (mClusterMaterialsBuf != nullptr) {
        GPU_storagebuf_free(mClusterMaterialsBuf);
    }
    mClusterMaterialsBuf = GPU_storagebuf_create_ex(cluster_materials.size() * sizeof(uint32_t),
                                                    cluster_materials.data(),
                                                    GPU_USAGE_STATIC,
                                                    "tessera_vis_cluster_materials");

    /* Instanced vertex-pulling batch: in_vid = 0 .. mMaxClusterIndices - 1 (local corner index
     * inside one cluster). One instance per visible cluster. */
    GPU_BATCH_DISCARD_SAFE(mForwardBatch);
    mForwardVbo = nullptr;
    mForwardIbo = nullptr;
    GPUVertFormat fwd_format = {};
    GPU_vertformat_attr_add(&fwd_format, "in_vid", gpu::VertAttrType::UINT_32);
    mForwardVbo = GPU_vertbuf_create_with_format_ex(fwd_format, GPU_USAGE_STATIC);
    GPU_vertbuf_data_alloc(*mForwardVbo, mMaxClusterIndices);
    std::vector<uint32_t> local_ids(mMaxClusterIndices);
    for (uint32_t i = 0; i < mMaxClusterIndices; ++i) {
        local_ids[i] = i;
    }
    GPU_vertbuf_attr_fill(mForwardVbo, 0, local_ids.data());
    mForwardBatch = GPU_batch_create_ex(
        GPU_PRIM_TRIS, mForwardVbo, nullptr, GPU_BATCH_OWNS_VBO);
}

void TesseraVisibilityPass::render_forward(gpu::StorageBuf *command_buf,
                                          size_t draw_count,
                                          const float4x4 &viewProj,
                                          const float4x4 &model_mat,
                                          const float3 &cam_pos,
                                          int viewport_width,
                                          int viewport_height,
                                          int shading_mode,
                                          int lighting_type,
                                          const float4 &base_color,
                                          const std::vector<uint32_t> *material_counts,
                                          uint32_t max_commands_per_material,
                                          gpu::StorageBuf *counter_buf)
{
    if (command_buf == nullptr || counter_buf == nullptr || draw_count == 0 ||
        mForwardBatch == nullptr || mPackedVertsBuf == nullptr || mIndicesBuf == nullptr ||
        mClusterMetaBuf == nullptr)
    {
        return;
    }
    if (viewport_width <= 0 || viewport_height <= 0) {
        return;
    }
    if (mForwardShader == nullptr) {
        mForwardShader = createForwardShader();
        if (mForwardShader == nullptr) {
            return;
        }
    }

    GPU_clip_control_unit_range(false);
    GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_color_mask(true, true, true, true);
    /* Two-sided foliage / open-card meshes must show their back faces (the
     * shader flips backfacing normals), and watertight geometry resolves
     * occlusion through depth testing, so face culling stays disabled for
     * the whole Tessera pipeline. */
    GPU_face_culling(GPU_CULL_NONE);
    GPU_blend(GPU_BLEND_NONE);

    /* Alpha cutout state: active when any material slot carries a diffuse
     * texture and mAlphaMask is enabled (foliage cards carry their silhouette in texture alpha). */
    bool alpha_mask_active = false;
    float4 has_tex_flags(0.0f);
    if (mAlphaMask) {
        for (int s = 0; s < 4; ++s) {
            const bool slot_has_diffuse = (s < int(mMaterials.size())) &&
                                          mMaterials[s].has_diffuse &&
                                          (mMaterials[s].diffuse != nullptr);
            if (slot_has_diffuse) {
                has_tex_flags[s] = 1.0f;
                alpha_mask_active = true;
            }
        }
    }

    GPU_shader_bind(mForwardShader);
    GPU_storagebuf_bind(command_buf, 0);
    GPU_storagebuf_bind(mClusterMetaBuf, 1);
    GPU_storagebuf_bind(mIndicesBuf, 2);
    GPU_storagebuf_bind(mPackedVertsBuf, 3);
    if (alpha_mask_active) {
        for (int s = 0; s < 4; ++s) {
            const bool slot_has_diffuse = (s < int(mMaterials.size())) &&
                                          mMaterials[s].has_diffuse &&
                                          (mMaterials[s].diffuse != nullptr);
            gpu::Texture *diff = slot_has_diffuse ?
                                     mMaterials[s].diffuse :
                                     ((mDefaultWhiteTex != nullptr) ? mDefaultWhiteTex : nullptr);
            if (diff != nullptr) {
                GPU_texture_bind(diff, s);
            }
        }
    }

    GPU_shader_uniform_mat4(mForwardShader, "view_proj", viewProj.ptr());
    GPU_shader_uniform_mat4(mForwardShader, "model_mat", model_mat.ptr());
    float3x3 fwd_norm = compute_normal_matrix(model_mat);
    GPU_shader_uniform_mat3(mForwardShader, "normal_mat", reinterpret_cast<const float(*)[3]>(fwd_norm.ptr()));
    GPU_shader_uniform_3f(mForwardShader, "camera_pos", cam_pos.x, cam_pos.y, cam_pos.z);
    /* Two-sided (foliage card) meshes render fully opaque in solid view. */
    float4 forward_base_color = base_color;
    if (mTwoSided) {
        forward_base_color.w = 1.0f;
    }
    GPU_shader_uniform_4f(mForwardShader, "base_color", forward_base_color.x, forward_base_color.y, forward_base_color.z, forward_base_color.w);
    GPU_shader_uniform_1i(mForwardShader, "shading_mode", shading_mode);
    GPU_shader_uniform_1i(mForwardShader, "lighting_type", lighting_type);
    GPU_shader_uniform_1i(mForwardShader, "cluster_count", int(mClusterCount));
    GPU_shader_uniform_1i(mForwardShader, "debug_clusters", shading_mode == TESSERA_SHADING_CLUSTERS ? 1 : 0);
    GPU_shader_uniform_1i(mForwardShader, "has_alpha_mask", alpha_mask_active ? 1 : 0);
    GPU_shader_uniform_4f(mForwardShader, "has_tex_flags", has_tex_flags.x, has_tex_flags.y, has_tex_flags.z, has_tex_flags.w);

    /* Single indirect instanced draw: counter_buf[72..75] holds one DrawArraysIndirectCommand
     * {max_cluster_indices, visible_cluster_count, 0, 0} written entirely on the GPU by the
     * traversal compute shader; command_buf holds the visible cluster indices. No CPU readback,
     * no multi-draw-indirect-count dependency (VKBatch does not implement it). */
    (void)material_counts;
    (void)max_commands_per_material;
    GPU_storagebuf_sync_as_indirect_buffer(counter_buf);
    GPU_batch_set_shader(mForwardBatch, mForwardShader);
    GPU_batch_draw_indirect(mForwardBatch, counter_buf, intptr_t(72 * sizeof(uint32_t)));
}

}  // namespace tessera
}  // namespace blender
