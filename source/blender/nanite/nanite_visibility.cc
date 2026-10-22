/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Phase 2: Visibility buffer rendering implementation.
 */

#include "nanite_visibility.hh"

#include "nanite_lod.hh"

#include "GPU_batch.hh"
#include "GPU_capabilities.hh"
#include "GPU_context.hh"
#include "GPU_index_buffer.hh"
#include "GPU_shader_create_info.hh"
#include "GPU_state.hh"
#include "GPU_vertex_buffer.hh"
#include "GPU_vertex_format.hh"

#include <cstring>

namespace blender {
namespace nanite {

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
  uint corner = gl_VertexID % 3u;
  uint idx = tri_indices[gl_VertexID];
  vec3 pos = vec3(positions[idx * 3u], positions[idx * 3u + 1u],
                  positions[idx * 3u + 2u]);
  gl_Position = view_proj * vec4(pos, 1.0);

  bary = vec2(corner == 0u ? 1.0 : 0.0, corner == 1u ? 1.0 : 0.0);
  tri_id = gl_VertexID / 3u;

  cluster_id = 0u;
  if (debug_clusters != 0) {
    for (uint c = 0u; c < uint(cluster_count); ++c) {
      uvec4 meta = cluster_meta[c];
      if (gl_VertexID >= meta.x && gl_VertexID < meta.x + meta.y) {
        cluster_id = c;
        break;
      }
    }
  }
}
)";

static const char *VIS_FRAG_SOURCE = R"(
void main()
{
  imageStore(vis_image, ivec2(gl_FragCoord.xy),
             uvec4(tri_id, packHalf2x16(bary), cluster_id, 0u));
}
)";

/* ---------------------------------------------------------------------------
 * Material pass: fetch the visibility record, reconstruct the attributes and
 * shade. The background is encoded with tri_id == 0xFFFFFFFF.
 * ------------------------------------------------------------------------- */

static const char *MATERIAL_VERT_SOURCE = R"(
void main()
{
  gl_Position = vec4(pos, 1.0);
}
)";

static const char *MATERIAL_FRAG_SOURCE = R"(
void main()
{
  ivec2 px = ivec2(gl_FragCoord.xy);
  uvec4 vis = texelFetch(vis_texture, px, 0);
  uint tri = vis.x;
  if (tri == 0xFFFFFFFFu) {
    out_color = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
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

  vec3 pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
  vec3 nor = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
  vec2 uv = bary.x * u0 + bary.y * u1 + bary.z * u2;

  if (shading_mode == 1u) {
    /* Cluster visualization: hash the cluster id to a flat color. */
    uint cid = vis.z;
    float hue = float((cid * 2654435761u) & 0xFFFFFFu) / float(0xFFFFFFu);
    out_color = vec4(fract(hue * 7.23), fract(hue * 3.71), fract(hue * 5.13), 1.0);
    return;
  }
  if (shading_mode == 2u) {
    out_color = vec4(nor * 0.5 + 0.5, 1.0);
    return;
  }

  /* Headlight diffuse with a touch of UV tint for orientation. */
  vec3 light_dir = normalize(camera_pos - pos);
  float diffuse = max(dot(nor, light_dir), 0.0);
  vec3 base = vec3(0.5, 0.5, 0.52) + vec3(0.2, 0.1, 0.0) * uv;
  out_color = vec4(base * (0.15 + 0.85 * diffuse), 1.0);
}
)";

/** Create the visibility-pass shader (vertex pulling + image store). */
gpu::Shader *createVisShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("nanite_vis");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;

    info.vertex_source("nanite_vis");
    info.vertex_source_generated = VIS_VERT_SOURCE;
    info.fragment_source("nanite_vis");
    info.fragment_source_generated = VIS_FRAG_SOURCE;

    /* Vertex pulling: no vertex attributes are consumed. */
    info.storage_buf(0, Qualifier::read, "uint", "tri_indices[]");
    info.storage_buf(1, Qualifier::read, "float", "positions[]");
    info.storage_buf(2, Qualifier::read, "uvec4", "cluster_meta[]");
    info.image(0,
               gpu::TextureFormat::UINT_32,
               Qualifier::write,
               ImageReadWriteType::Uint2D,
               "vis_image");

    info.push_constant(Type::float4x4_t, "view_proj");
    info.push_constant(Type::int_t, "debug_clusters");
    info.push_constant(Type::int_t, "cluster_count");

    StageInterfaceInfo iface("nanite_vis_iface", "");
    iface.flat(Type::uint_t, "tri_id");
    iface.smooth(Type::float2_t, "bary");
    iface.flat(Type::uint_t, "cluster_id");
    info.vertex_out(iface);

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info(raw);
}

/** Create the material-pass shader (fullscreen shading). */
gpu::Shader *createMaterialShader()
{
    using namespace blender::gpu::shader;

    ShaderCreateInfo info("nanite_material");
    info.builtins_ |= BuiltinBits::NO_BUFFER_TYPE_LINTING;

    info.vertex_source("nanite_material");
    info.vertex_source_generated = MATERIAL_VERT_SOURCE;
    info.fragment_source("nanite_material");
    info.fragment_source_generated = MATERIAL_FRAG_SOURCE;

    info.vertex_in(0, Type::float3_t, "pos");
    info.fragment_out(0, Type::float4_t, "out_color");

    info.storage_buf(0, Qualifier::read, "float", "positions[]");
    info.storage_buf(1, Qualifier::read, "float", "normals[]");
    info.storage_buf(2, Qualifier::read, "float", "uvs[]");
    info.storage_buf(3, Qualifier::read, "uint", "tri_indices[]");
    info.sampler(0, ImageType::Uint2D, "vis_texture");

    info.push_constant(Type::float3_t, "camera_pos");
    info.push_constant(Type::int_t, "shading_mode");

    const GPUShaderCreateInfo *raw = reinterpret_cast<const GPUShaderCreateInfo *>(&info);
    return GPU_shader_create_from_info(raw);
}

}  // namespace

NaniteVisibilityPass::~NaniteVisibilityPass()
{
    terminate();
}

bool NaniteVisibilityPass::init()
{
    if (mVisShader != nullptr) {
        return true;
    }
    if (!gpu_available()) {
        return false;
    }
    mVisShader = createVisShader();
    mMaterialShader = createMaterialShader();
    if (mVisShader == nullptr || mMaterialShader == nullptr) {
        return false;
    }

    /* Full-screen triangle covering NDC. */
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

    /* Vertex-pulling batch: a dummy attribute buffer with one entry per
     * index-buffer position (values are never read; gl_VertexID drives the
     * fetch). Sized when the mesh data is staged. */
    GPUVertFormat idx_format = {};
    GPU_vertformat_attr_add(&idx_format, "vid", gpu::VertAttrType::UINT_32);
    mIndexVbo = GPU_vertbuf_create_with_format(idx_format);
    mMeshBatch = GPU_batch_create_ex(
        GPU_PRIM_TRIS, mIndexVbo, nullptr, GPU_BATCH_OWNS_VBO);

    return true;
}

void NaniteVisibilityPass::terminate()
{
    GPU_FRAMEBUFFER_FREE_SAFE(mFrameBuffer);
    GPU_TEXTURE_FREE_SAFE(mVisTexture);
    GPU_TEXTURE_FREE_SAFE(mDepthTexture);
    GPU_BATCH_DISCARD_SAFE(mFullscreenBatch);
    GPU_VERTBUF_DISCARD_SAFE(mFullscreenVbo);
    GPU_BATCH_DISCARD_SAFE(mMeshBatch);
    GPU_VERTBUF_DISCARD_SAFE(mIndexVbo);
    GPU_SHADER_FREE_SAFE(mVisShader);
    GPU_SHADER_FREE_SAFE(mMaterialShader);
    if (mPositionsBuf != nullptr) { GPU_storagebuf_free(mPositionsBuf); mPositionsBuf = nullptr; }
    if (mNormalsBuf != nullptr) { GPU_storagebuf_free(mNormalsBuf); mNormalsBuf = nullptr; }
    if (mUvsBuf != nullptr) { GPU_storagebuf_free(mUvsBuf); mUvsBuf = nullptr; }
    if (mIndicesBuf != nullptr) { GPU_storagebuf_free(mIndicesBuf); mIndicesBuf = nullptr; }
    if (mClusterMetaBuf != nullptr) { GPU_storagebuf_free(mClusterMetaBuf); mClusterMetaBuf = nullptr; }
    mVertexData = NaniteGpuVertexData();
    mVertexCount = 0;
    mIndexCount = 0;
    mClusterCount = 0;
    mWidth = 0;
    mHeight = 0;
}

bool NaniteVisibilityPass::ensure_targets(int width, int height)
{
    if (mVisTexture != nullptr && mWidth == width && mHeight == height) {
        return true;
    }
    GPU_TEXTURE_FREE_SAFE(mVisTexture);
    GPU_TEXTURE_FREE_SAFE(mDepthTexture);
    GPU_FRAMEBUFFER_FREE_SAFE(mFrameBuffer);
    mWidth = width;
    mHeight = height;

    mVisTexture = GPU_texture_create_2d("nanite_vis_buffer",
                                        width,
                                        height,
                                        1,
                                        gpu::TextureFormat::UINT_32_32_32_32,
                                        GPU_TEXTURE_USAGE_GENERAL,
                                        nullptr);
    mDepthTexture = GPU_texture_create_2d("nanite_vis_depth",
                                          width,
                                          height,
                                          1,
                                          gpu::TextureFormat::SFLOAT_32_DEPTH,
                                          GPU_TEXTURE_USAGE_GENERAL,
                                          nullptr);
    if (mVisTexture == nullptr || mDepthTexture == nullptr) {
        return false;
    }
    mFrameBuffer = GPU_framebuffer_create("nanite_vis_fb");
    GPU_framebuffer_ensure_config(&mFrameBuffer,
                                  {GPU_ATTACHMENT_TEXTURE(mDepthTexture),
                                   GPU_ATTACHMENT_NONE});
    return mFrameBuffer != nullptr;
}

bool NaniteVisibilityPass::upload_dirty_data()
{
    const size_t index_count = mVertexData.indices.size();
    if (index_count == 0) {
        return false;
    }
    if (mIndicesBuf == nullptr || index_count != mIndexCount ||
        mVertexData.vertex_count != mVertexCount)
    {
        if (mPositionsBuf != nullptr) { GPU_storagebuf_free(mPositionsBuf); mPositionsBuf = nullptr; }
        if (mNormalsBuf != nullptr) { GPU_storagebuf_free(mNormalsBuf); mNormalsBuf = nullptr; }
        if (mUvsBuf != nullptr) { GPU_storagebuf_free(mUvsBuf); mUvsBuf = nullptr; }
        if (mIndicesBuf != nullptr) { GPU_storagebuf_free(mIndicesBuf); mIndicesBuf = nullptr; }
        mPositionsBuf = GPU_storagebuf_create_ex(
            mVertexData.positions.size() * sizeof(float),
            nullptr,
            GPU_USAGE_DYNAMIC,
            "nanite_positions");
        mNormalsBuf = GPU_storagebuf_create_ex(mVertexData.normals.size() *
                                                   sizeof(float),
                                               nullptr,
                                               GPU_USAGE_DYNAMIC,
                                               "nanite_normals");
        mUvsBuf = GPU_storagebuf_create_ex(mVertexData.uvs.size() * sizeof(float),
                                           nullptr,
                                           GPU_USAGE_DYNAMIC,
                                           "nanite_uvs");
        mIndicesBuf = GPU_storagebuf_create_ex(index_count * sizeof(uint32_t),
                                               nullptr,
                                               GPU_USAGE_DYNAMIC,
                                               "nanite_tri_indices");
        if (mPositionsBuf == nullptr || mNormalsBuf == nullptr ||
            mUvsBuf == nullptr || mIndicesBuf == nullptr)
        {
            return false;
        }
        mVertexCount = mVertexData.vertex_count;
        mIndexCount = index_count;

        /* Grow the vertex-pulling batch (one dummy attribute per corner). */
        GPU_vertbuf_data_alloc(*mIndexVbo, uint(index_count));
        std::vector<uint32_t> identity(index_count);
        for (size_t i = 0; i < index_count; ++i) {
            identity[i] = uint32_t(i);
        }
        GPU_vertbuf_attr_fill(mIndexVbo, 0, identity.data());
    }
    GPU_storagebuf_update(mPositionsBuf, mVertexData.positions.data());
    GPU_storagebuf_update(mNormalsBuf, mVertexData.normals.data());
    GPU_storagebuf_update(mUvsBuf, mVertexData.uvs.data());
    GPU_storagebuf_update(mIndicesBuf, mVertexData.indices.data());
    return true;
}

void NaniteVisibilityPass::render(gpu::StorageBuf *command_buf,
                                  size_t draw_count,
                                  const float4x4 &viewProj,
                                  const float3 &cam_pos,
                                  int viewport_width,
                                  int viewport_height,
                                  int shading_mode)
{
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

    /* Encode "no geometry" as tri_id == 0xFFFFFFFF. */
    const uint32_t clear_value[4] = {0xFFFFFFFFu, 0u, 0u, 0u};
    GPU_texture_clear(
        mVisTexture, GPU_DATA_UINT, static_cast<const void *>(clear_value));

    /* ---------------- Pass 1: visibility ---------------- */
    GPU_framebuffer_bind(mFrameBuffer);
    GPU_framebuffer_clear_depth(mFrameBuffer, 1.0f);
    GPU_depth_test(GPU_DEPTH_LESS_EQUAL);
    GPU_depth_mask(true);
    GPU_face_culling(GPU_CULL_NONE);
    GPU_blend(GPU_BLEND_NONE);
    GPU_color_mask(false, false, false, false);

    GPU_shader_bind(mVisShader);
    GPU_storagebuf_bind(mIndicesBuf, 0);
    GPU_storagebuf_bind(mPositionsBuf, 1);
    if (mClusterMetaBuf != nullptr) {
        GPU_storagebuf_bind(mClusterMetaBuf, 2);
    }
    GPU_texture_image_bind(mVisTexture, 0);
    GPU_shader_uniform_mat4(mVisShader, "view_proj", viewProj.ptr());
    GPU_shader_uniform_1i(
        mVisShader,
        "debug_clusters",
        shading_mode == NANITE_SHADING_CLUSTERS ? 1 : 0);
    GPU_shader_uniform_1i(mVisShader, "cluster_count", int(mClusterCount));

    GPU_batch_multi_draw_indirect(
        mMeshBatch, command_buf, int(draw_count), 0, sizeof(uint32_t) * 4);

    GPU_color_mask(true, true, true, true);
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

    /* ---------------- Pass 2: material ---------------- */
    GPU_framebuffer_bind(previous_fb != nullptr ? previous_fb : mFrameBuffer);
    GPU_depth_test(GPU_DEPTH_NONE);
    GPU_blend(GPU_BLEND_NONE);

    GPU_shader_bind(mMaterialShader);
    GPU_storagebuf_bind(mPositionsBuf, 0);
    GPU_storagebuf_bind(mNormalsBuf, 1);
    GPU_storagebuf_bind(mUvsBuf, 2);
    GPU_storagebuf_bind(mIndicesBuf, 3);
    GPU_texture_bind(mVisTexture, 0);
    GPU_shader_uniform_3f(
        mMaterialShader, "camera_pos", cam_pos.x, cam_pos.y, cam_pos.z);
    GPU_shader_uniform_1i(mMaterialShader, "shading_mode", shading_mode);

    GPU_batch_draw(mFullscreenBatch);
}

void NaniteVisibilityPass::set_data(const NaniteData &data)
{
    if (!gpu_available()) {
        return;
    }
    const size_t vertex_count = data.positions.size();
    const size_t index_count = data.indices.size();
    const size_t cluster_count = data.clusters.size();
    if (vertex_count == mVertexCount && index_count == mIndexCount &&
        cluster_count == mClusterCount)
    {
        return;
    }
    mVertexCount = vertex_count;
    mIndexCount = index_count;
    mClusterCount = cluster_count;

    flattenVertexDataForGpu(data, mVertexData);

    if (mClusterMetaBuf != nullptr) { GPU_storagebuf_free(mClusterMetaBuf); mClusterMetaBuf = nullptr; }
    NaniteGpuClusterData cluster_data;
    flattenClustersForGpu(data, cluster_data);
    mClusterMetaBuf = GPU_storagebuf_create_ex(cluster_data.meta.size() *
                                                   sizeof(uint32_t),
                                               cluster_data.meta.data(),
                                               GPU_USAGE_DYNAMIC,
                                               "nanite_cluster_meta");
    upload_dirty_data();
}

}  // namespace nanite
}  // namespace blender

