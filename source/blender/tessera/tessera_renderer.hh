/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  GPU renderer: phase 1 — GPU-driven traversal, indirect multi-draw.
 *
 *  The renderer owns the indirect draw command storage buffer, the per-cluster
 *  data buffers and the traversal compute shader. Draw calls are issued
 *  through Blender's GPU batch abstraction (GPU_batch_multi_draw_indirect).
 */

#pragma once

#include "tessera.h"
#include "tessera_gpu_traverse.hh"
#include "tessera_visibility.hh"

#include "GPU_batch.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace blender {
namespace tessera {

/**
 * GPU renderer: phases 1 + 2 — GPU-driven traversal and visibility-buffer
 * shading.
 *
 * Every frame a single compute-shader dispatch performs frustum culling and
 * LOD selection for all clusters in parallel and atomically appends indirect
 * draw commands into a storage buffer. The only CPU work left is one
 * dispatch, one 4-byte counter readback and (only when the mesh changes) one
 * cluster-data upload. The draw step is the two-pass visibility-buffer
 * pipeline (visibility rasterization + full-screen material pass); no
 * caller-supplied batch is needed.
 *
 * When compute shaders are unavailable the renderer transparently falls back
 * to the validated CPU traversal (selectClustersFlat) and uploads the result.
 */
class TesseraRenderer
{
public:
    TesseraRenderer() = default;
    ~TesseraRenderer();

    TesseraRenderer(const TesseraRenderer&) = delete;
    TesseraRenderer& operator=(const TesseraRenderer&) = delete;

    /** Initialize the renderer (create the indirect draw buffer). */
    bool init();

    /** Release GPU resources. */
    void terminate();

    /** Call once per frame to recompute LOD selection and update the command buffer. */
    void update(const TesseraData &data,
                const float4x4 &viewProj,
                float projScaleY,
                float viewportHeight,
                const float4x4 *lodViewProj = nullptr,
                const float *lodProjScaleY = nullptr);

    /** Override the LOD pixel error threshold for the next frames. */
    void setPixelError(float pixelError) { mPixelError = pixelError; }
    float getPixelError() const { return mPixelError; }

    /** Select the material-pass shading mode (diffuse/clusters/normals). */
    void setShadingMode(TesseraShadingMode mode) { mShadingMode = mode; }

    /** Configure reverse-Z for engines that use inverted depth (EEVEE). */
    void setReverseZ(bool enabled) { mReverseZ = enabled; }
    bool getReverseZ() const { return mReverseZ; }

    void set_material_textures(const std::vector<TesseraMaterialTextures> &materials)
    {
        mVisibilityPass.set_material_textures(materials);
    }

    /**
     * Call once per frame to render the selected clusters into the currently
     * bound framebuffer with the visibility-buffer pipeline.
     */
    void draw(int viewportWidth, int viewportHeight);

    /**
     * Depth-only prepass: render clusters into the depth buffer with Reverse-Z.
     */
    void draw_prepass(int viewportWidth, int viewportHeight);

    /**
     * G-Buffer deferred pass: render clusters into EEVEE G-Buffer MRT targets.
     */
    void draw_gbuffer(int viewportWidth, int viewportHeight, const float4x4 &object_model = float4x4::identity());

    size_t drawCount(uint32_t material_index = 0) const
    {
        if (mMaterialDrawCounts.size() > material_index) {
            return mMaterialDrawCounts[material_index];
        }
        return (material_index == 0) ? mDrawCount : 0;
    }

    size_t totalDrawCount() const { return mDrawCount; }

    size_t materialCommandOffset(uint32_t material_index) const
    {
        return size_t(material_index) * mMaxCommandsPerMaterial;
    }

    const std::vector<uint32_t> &materialDrawCounts() const { return mMaterialDrawCounts; }
    uint32_t maxCommandsPerMaterial() const { return mMaxCommandsPerMaterial; }

    /** Diagnostic helper: count active clusters per LOD level. */
    void getLevelCounts(const TesseraData &data,
                        const float4x4 &lodViewProj,
                        float projScaleY,
                        float viewportHeight,
                        std::map<uint32_t, uint32_t> &out_counts) const;

    /** Realtime Viewport Telemetry */
    void setTelemetryEnabled(bool enabled) { mTelemetryEnabled = enabled; }
    bool isTelemetryEnabled() const { return mTelemetryEnabled; }
    const TesseraTelemetry &getTelemetry() const { return mTelemetry; }

    /** True when the GPU compute traversal is in use. */
    bool usesGpuTraversal() const { return mGpuTraversalActive; }

private:
    bool mTelemetryEnabled = true;
    TesseraTelemetry mTelemetry;

    /** CPU fallback: run the flat traversal and upload the commands. */
    void updateWithCpuTraversal(const TesseraData &data,
                                const float4x4 &viewProj,
                                float projScaleY,
                                float viewportHeight);

    /** GPU path: upload cluster data (if changed), dispatch, read back count. */
    void updateWithGpuTraversal(const TesseraData &data,
                                const float4x4 &viewProj,
                                float projScaleY,
                                float viewportHeight);

    /** (Re)create or resize one SSBO to at least \a byteSize. */
    bool ensureStorageBuf(gpu::StorageBuf **buf, size_t byteSize, const char *name);

    // Staging host-side command list produced by the CPU fallback path.
    std::vector<TesseraDrawCommand> mPendingCommands;
    // DrawArraysIndirectCommand stream for the CPU fallback upload.
    std::vector<uint32_t> mCpuCommandStream;

    // GPU storage buffer holding the indirect draw commands (std430 layout).
    gpu::StorageBuf *mDrawBuffer = nullptr;

    // Compute shader implementing the traversal (created lazily).
    gpu::Shader *mTraverseShader = nullptr;

    // Visibility-buffer pipeline (phase 2).
    TesseraVisibilityPass mVisibilityPass;
    bool mVisibilityInitialized = false;

    // Per-cluster data buffers (uploaded only when the mesh changes).
    gpu::StorageBuf *mSphereBuf = nullptr;
    gpu::StorageBuf *mParentSphereBuf = nullptr;
    gpu::StorageBuf *mErrorsBuf = nullptr;
    gpu::StorageBuf *mMetaBuf = nullptr;
    gpu::StorageBuf *mMaterialsBuf = nullptr;
    // Shared atomic counter written by the compute shader, read back on CPU.
    gpu::StorageBuf *mCounterBuf = nullptr;

    TesseraGpuClusterData mGpuClusters;

    // Fingerprint of the last uploaded cluster data (dirty tracking).
    size_t mUploadedClusterCount = size_t(-1);
    size_t mUploadedIndexCount = size_t(-1);
    float mUploadedMaxParentError = -1.0f;

    // Number of draw calls written this frame.
    size_t mDrawCount = 0;
    static constexpr size_t kMaxMaterials = 64;
    uint32_t mMaxMaterials = 1;
    uint32_t mMaxCommandsPerMaterial = 2000000;
    std::vector<uint32_t> mMaterialDrawCounts;

    // Data/matrix used by the last update (consumed by the draw step).
    const TesseraData *mLastData = nullptr;
    float4x4 mLastViewProj = float4x4::identity();
    float4x4 mLastLodViewProj = float4x4::identity();
    float mLastProjScaleY = 1.0f;
    float mLastLodProjScaleY = 1.0f;
    float mLastViewportHeight = 0.0f;
    float mLastPixelError = -1.0f;
    bool mMatricesCurrent = false;  // track if matrices have been updated this frame

    // LOD pixel error threshold (pixels).
    float mPixelError = 1.0f;

    // Material-pass shading mode.
    TesseraShadingMode mShadingMode = TESSERA_SHADING_DIFFUSE;

    // Inverted depth support (EEVEE Reverse-Z).
    bool mReverseZ = false;

    // Set once the GPU path is proven working; set permanently if the
    // compute shader could not be created.
    bool mGpuTraversalActive = false;
    bool mGpuPathFailed = false;

    // Constant host-side capacity. Granularity is kept to avoid reallocation.
    static constexpr size_t kDrawCommandCapacity = 2000000;
};

}  // namespace tessera
}  // namespace blender
