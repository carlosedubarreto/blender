/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 2: Visibility buffer rendering.
 *
 *  Two-pass deferred pipeline:
 *
 *  1. Visibility pass: hardware-rasterized indirect multi-draw (one
 *     non-indexed draw command per cluster, vertex pulling through SSBOs).
 *     The fragment shader writes a per-pixel visibility record
 *     (global triangle id + perspective-correct barycentric weights +
 *     cluster id) into an integer texture, with depth testing against the
 *     pass depth buffer.
 *
 *  2. Material pass: a full-screen triangle fetches the visibility record,
 *     pulls the triangle attributes from SSBOs, interpolates with the
 *     stored barycentric weights and shades. Cluster-color and normal
 *     visualization modes are built in.
 *
 *  No CPU work per frame beyond the single indirect multi-draw and the
 *  full-screen draw; vertex/index/cluster buffers stay resident.
 */

#pragma once

#include "tessera.h"
#include "tessera_gpu_data.hh"

#include "GPU_batch.hh"
#include "GPU_framebuffer.hh"
#include "GPU_index_buffer.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_texture.hh"
#include "GPU_vertex_buffer.hh"

#include <cstddef>
#include <cstdint>

namespace blender {
namespace tessera {

/** Material-pass shading modes. */
enum TesseraShadingMode {
    TESSERA_SHADING_DIFFUSE = 0,
    TESSERA_SHADING_CLUSTERS = 1,
    TESSERA_SHADING_NORMALS = 2,
    TESSERA_SHADING_TANGENTS = 3,
    TESSERA_SHADING_UVS = 4,
};

/**
 * Workbench studio/matcap lighting parameters packed for the forward
 * shader's `workbench_light[]` storage buffer (14 vec4 entries, layout
 * documented in FORWARD_FRAG_SOURCE). Vectors follow Workbench's own
 * convention: view space (matching #WorldData in
 * workbench_shader_shared.hh), so the forward pass must rotate its
 * world-space normals/positions with the same default-view rotation the
 * native deferred pipeline uses.
 */
struct TesseraWorkbenchLight {
    float4 light_dir[4]; /** xyz: studio light direction (view space) */
    float4 diffuse_color_wrap[4]; /** rgb: diffuse color, a: wrap factor */
    float4 specular_color[4]; /** rgb: specular color */
    float4 material; /** x: roughness, y: metallic, z: use_specular, w: matcap flip X */
    float4 ambient; /** rgb: ambient color, w: 1 = perspective camera */
};

/**
 * Renders the selected clusters with a visibility buffer, then shades.
 * The cluster/vertex data is owned by the caller's TesseraRenderer; this
 * pass receives the indirect draw-command buffer and the draw count.
 */
class TesseraVisibilityPass
{
 public:
    TesseraVisibilityPass() = default;
    ~TesseraVisibilityPass();

    TesseraVisibilityPass(const TesseraVisibilityPass&) = delete;
    TesseraVisibilityPass& operator=(const TesseraVisibilityPass&) = delete;

    /** Create shaders and the full-screen batch. Safe to call once. */
    bool init();

    void terminate();

    /** Stage cluster/vertex data (uploads when the fingerprint changes). */
    void set_data(const TesseraData &data);

    /**
     * Run the two-pass render into the currently bound framebuffer.
     * \param command_buf indirect draw commands (DrawArraysIndirectCommand
     *                    layout, 16 bytes per command).
     * \param draw_count number of valid commands.
     */
    void render(gpu::StorageBuf *command_buf,
                size_t draw_count,
                const float4x4 &viewProj,
                const float3 &cam_pos,
                int viewport_width,
                int viewport_height,
                int shading_mode,
                const float4x4 &object_model = float4x4::identity(),
                bool reverse_z = false,
                const std::vector<uint32_t> *material_counts = nullptr,
                uint32_t max_commands_per_material = 0,
                gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Render depth-only prepass into the currently bound framebuffer.
     */
    void render_depth_prepass(gpu::StorageBuf *command_buf,
                             size_t draw_count,
                             const float4x4 &viewProj,
                             int viewport_width,
                             int viewport_height,
                             bool reverse_z = false,
                             const std::vector<uint32_t> *material_counts = nullptr,
                             uint32_t max_commands_per_material = 0,
                             gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Render G-Buffer pass into the currently bound EEVEE G-Buffer framebuffer.
     */
    void render_gbuffer(gpu::StorageBuf *command_buf,
                        size_t draw_count,
                        const float4x4 &viewProj,
                        const float3 &cam_pos,
                        int viewport_width,
                        int viewport_height,
                        const float4x4 &object_model = float4x4::identity(),
                        bool reverse_z = true,
                        const std::vector<uint32_t> *material_counts = nullptr,
                        uint32_t max_commands_per_material = 0,
                        gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Direct single-pass forward rasterization for Solid Mode (Workbench).
     * Bypasses the visibility buffer resolve entirely and uses hardware
     * indexed drawing (GL_ELEMENT_ARRAY_BUFFER).
     */
    void render_forward(gpu::StorageBuf *command_buf,
                        size_t draw_count,
                        const float4x4 &viewProj,
                        const float4x4 &model_mat,
                        const float3 &cam_pos,
                        int viewport_width,
                        int viewport_height,
                        int shading_mode = 0,
                        int lighting_type = 1,
                        const float4 &base_color = float4(0.8f, 0.8f, 0.8f, 1.0f),
                        const std::vector<uint32_t> *material_counts = nullptr,
                        uint32_t max_commands_per_material = 0,
                        bool reverse_z = false,
                        gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Instanced variant of #render_forward for scattered instance fields:
     * every surviving object instance of one archetype is rasterized with a
     * SINGLE indirect instanced draw. The per-instance world matrices are
     * uploaded to a storage buffer; the vertex shader decodes
     * gl_InstanceID -> (object instance, visible cluster slot) using the
     * visible-cluster count the traversal finalize pass stored at
     * draw_counters[77]. Requires #TesseraRenderer::update to have been
     * called with the same instance multiplier so the GPU-written
     * DrawArraysIndirectCommand instanceCount equals
     * visible_clusters * instance_count.
     *
     * \param viewProj pure world->clip projection (NO instance/object matrix
     *                 baked in — the per-instance model matrix comes from the
     *                 instance buffer).
     * \param cam_pos  camera position in world space (for studio lighting).
     */
    void render_forward_instances(gpu::StorageBuf *command_buf,
                                  size_t draw_count,
                                  const float4x4 *instance_mats,
                                  size_t instance_count,
                                  const float4x4 &viewProj,
                                  const float3 &cam_pos,
                                  int viewport_width,
                                  int viewport_height,
                                  int shading_mode = 0,
                                  int lighting_type = 1,
                                  const float4 &base_color = float4(0.8f, 0.8f, 0.8f, 1.0f),
                                  bool reverse_z = false,
                                  gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Direct instanced G-Buffer MRT rasterization for scattered fields in
     * EEVEE Next (Material Preview & Render View). Draws all visible clusters
     * across all instances directly into EEVEE's G-Buffer framebuffer in one
     * indirect multi-draw call, eliminating fullscreen visibility passes.
     */
    void render_gbuffer_instances(gpu::StorageBuf *command_buf,
                                  size_t draw_count,
                                  const float4x4 *instance_mats,
                                  size_t instance_count,
                                  const float4x4 &viewProj,
                                  const float3 &cam_pos,
                                  int viewport_width,
                                  int viewport_height,
                                  bool reverse_z = false,
                                  gpu::StorageBuf *counter_buf = nullptr);

    /**
     * Direct instanced Reverse-Z depth prepass rasterization for scattered
     * instance fields in EEVEE Next. Writes depth for all visible clusters
     * across instances directly into EEVEE's prepass framebuffer with alpha cutout.
     */
    void render_depth_instances(gpu::StorageBuf *command_buf,
                                size_t draw_count,
                                const float4x4 *instance_mats,
                                size_t instance_count,
                                const float4x4 &viewProj,
                                const float3 &cam_pos,
                                int viewport_width,
                                int viewport_height,
                                bool reverse_z = false,
                                gpu::StorageBuf *counter_buf = nullptr);

    void set_material_textures(const std::vector<TesseraMaterialTextures> &materials)
    {
        mMaterials = materials;
    }

    /**
     * Two-sided / foliage-card mode: the geometry consists of open
     * single-sided leaf/grass cards that must show their back faces with
     * flipped normals. Face culling stays disabled either way (occlusion is
     * resolved by depth), and the solid forward path renders such meshes
     * fully opaque.
     */
    void set_two_sided(bool two_sided)
    {
        mTwoSided = two_sided;
    }

    /**
     * Alpha cutout: reject fragments whose base color texture alpha falls
     * below the cutout threshold (foliage card surrounds). Only meaningful
     * when diffuse textures are bound.
     */
    void set_alpha_mask(bool enabled)
    {
        mAlphaMask = enabled;
    }

    /**
     * Exact Workbench studio/matcap lighting for the forward solid pass
     * (color parity with the native deferred Workbench pipeline).
     * \param light studio light data (view-space, straight from #WorldData)
     * \param studio_match when set, the forward shader evaluates the exact
     *                      ported Workbench shading model instead of its
     *                      cheap headlight approximation (EEVEE Next never
     *                      sets this).
     */
    void set_workbench_lighting(const TesseraWorkbenchLight &light, bool studio_match)
    {
        mWorkbenchLight = light;
        mStudioMatch = studio_match;
    }

    /** World->view rotation of the default view (float3x3(viewmat())). */
    void set_view_rotation(const float3x3 &view_rot)
    {
        mViewRot = view_rot;
    }

    /** Workbench matcap texture: 2-layer float array (diffuse, specular). */
    void set_matcap(gpu::Texture *matcap_tex)
    {
        mMatcapTex = matcap_tex;
    }

    void setTextureScale(float scale)
    {
        mTextureScale = std::clamp(scale, 0.05f, 1.0f);
    }

    float getTextureScale() const
    {
        return mTextureScale;
    }

 private:
    float mTextureScale = 1.0f;
    bool mTwoSided = false;
    bool mAlphaMask = false;
    bool ensure_targets(int width, int height);
    bool upload_dirty_data();
    bool ensure_vis_resources();

    /**
     * Shared tail of #render_forward / #render_forward_instances: binds the
     * forward shader, its storage buffers (cluster list, metadata, indices,
     * packed vertices, instance matrices, forward args), samplers and push
     * constants, then issues the single indirect instanced draw.
     * \param instance_count 0 = single-matrix mode (model_mat/normal_mat used);
     *                        > 0 = per-instance matrices from #mInstanceMatsBuf.
     */
    void forward_draw_common(gpu::StorageBuf *command_buf,
                             gpu::StorageBuf *counter_buf,
                             const float4x4 &viewProj,
                             const float4x4 &model_mat,
                             const float3x3 &normal_mat,
                             const float3 &cam_pos,
                             int viewport_width,
                             int viewport_height,
                             int shading_mode,
                             int lighting_type,
                             const float4 &base_color,
                             const float4 &has_tex_flags,
                             bool alpha_mask_active,
                             bool reverse_z,
                             size_t instance_count);

    // Shaders (created lazily with an active context).
    gpu::Shader *mVisShader = nullptr;
    gpu::Shader *mMaterialShader = nullptr;
    gpu::Shader *mDepthPrepassShader = nullptr;
    gpu::Shader *mDepthPrepassAlphaShader = nullptr;
    gpu::Shader *mGbufferShader = nullptr;
    gpu::Shader *mForwardShader = nullptr;
    gpu::Shader *mForwardGbufferShader = nullptr;
    gpu::Shader *mForwardDepthShader = nullptr;

    // Visibility targets.
    gpu::Texture *mVisTexture = nullptr;
    gpu::Texture *mDepthTexture = nullptr;

    // Object model matrix for transforming object-space vertices to world space.
    float4x4 object_model = float4x4::identity();
    gpu::FrameBuffer *mFrameBuffer = nullptr;
    int mWidth = 0;
    int mHeight = 0;

    // Full-screen triangle for the material pass.
    gpu::VertBuf *mFullscreenVbo = nullptr;
    gpu::Batch *mFullscreenBatch = nullptr;

    // Vertex-pulling batch: one dummy attribute, no index buffer.
    gpu::VertBuf *mIndexVbo = nullptr;
    gpu::Batch *mMeshBatch = nullptr;

    // Instanced vertex-pulling batch for Solid Mode forward rasterization.
    gpu::VertBuf *mForwardVbo = nullptr;
    gpu::IndexBuf *mForwardIbo = nullptr;
    gpu::Batch *mForwardBatch = nullptr;

    // Vertex/index data buffers (uploaded when the data changes).
    gpu::StorageBuf *mPackedVertsBuf = nullptr;
    gpu::StorageBuf *mPositionsBuf = nullptr;
    gpu::StorageBuf *mNormalsBuf = nullptr;
    gpu::StorageBuf *mUvsBuf = nullptr;
    gpu::StorageBuf *mTangentsBuf = nullptr;
    gpu::StorageBuf *mIndicesBuf = nullptr;
    gpu::StorageBuf *mClusterMetaBuf = nullptr;
    gpu::StorageBuf *mClusterMaterialsBuf = nullptr;
    gpu::StorageBuf *mTriMaterialsBuf = nullptr;
    gpu::StorageBuf *mMaterialParamsBuf = nullptr;

    // Per-object-instance world matrices for scattered fields (grown as needed).
    gpu::StorageBuf *mInstanceMatsBuf = nullptr;
    size_t mInstanceMatsCapacity = 0;
    // 1-matrix fallback bound when drawing in single (non-instanced) mode.
    gpu::StorageBuf *mDummyInstanceBuf = nullptr;

    // Default fallback textures for unused material slots.
    gpu::Texture *mDefaultWhiteTex = nullptr;
    gpu::Texture *mDefaultNormalTex = nullptr;
    gpu::Texture *mDefaultRoughTex = nullptr;
    // 2-layer 1x1 fallback for the matcap sampler (studio / no matcap bound).
    gpu::Texture *mDefaultMatcapTex = nullptr;

    // Workbench studio/matcap parity state (see set_workbench_lighting()).
    TesseraWorkbenchLight mWorkbenchLight = {};
    bool mStudioMatch = false;
    float3x3 mViewRot = float3x3::identity();
    gpu::Texture *mMatcapTex = nullptr;
    gpu::StorageBuf *mWorkbenchLightBuf = nullptr;

    std::vector<TesseraMaterialTextures> mMaterials;

    const TesseraData *mCachedVisData = nullptr;
    bool mVisResourcesDirty = false;
    uint32_t mMaxClusterIndices = 384;
    TesseraGpuVertexData mVertexData;
    size_t mVertexCount = 0;
    size_t mIndexCount = 0;
    size_t mClusterCount = 0;
    float mBoundsMin[3] = {0.0f, 0.0f, 0.0f};
    float mBoundsMax[3] = {0.0f, 0.0f, 0.0f};
};

}  // namespace tessera
}  // namespace blender

