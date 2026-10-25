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
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_texture.hh"

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
                uint32_t max_commands_per_material = 0);

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
                             uint32_t max_commands_per_material = 0);

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
                        uint32_t max_commands_per_material = 0);

    void set_material_textures(const std::vector<TesseraMaterialTextures> &materials)
    {
        mMaterials = materials;
    }

 private:
    bool ensure_targets(int width, int height);
    bool upload_dirty_data();

    // Shaders (created lazily with an active context).
    gpu::Shader *mVisShader = nullptr;
    gpu::Shader *mMaterialShader = nullptr;
    gpu::Shader *mDepthPrepassShader = nullptr;
    gpu::Shader *mGbufferShader = nullptr;

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

    // Vertex/index data buffers (uploaded when the data changes).
    gpu::StorageBuf *mPositionsBuf = nullptr;
    gpu::StorageBuf *mNormalsBuf = nullptr;
    gpu::StorageBuf *mUvsBuf = nullptr;
    gpu::StorageBuf *mTangentsBuf = nullptr;
    gpu::StorageBuf *mIndicesBuf = nullptr;
    gpu::StorageBuf *mClusterMetaBuf = nullptr;
    gpu::StorageBuf *mTriMaterialsBuf = nullptr;
    gpu::StorageBuf *mMaterialParamsBuf = nullptr;

    // Default fallback textures for unused material slots.
    gpu::Texture *mDefaultWhiteTex = nullptr;
    gpu::Texture *mDefaultNormalTex = nullptr;
    gpu::Texture *mDefaultRoughTex = nullptr;

    std::vector<TesseraMaterialTextures> mMaterials;

    TesseraGpuVertexData mVertexData;
    size_t mVertexCount = 0;
    size_t mIndexCount = 0;
    size_t mClusterCount = 0;
};

}  // namespace tessera
}  // namespace blender
