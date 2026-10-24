#pragma once

/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Public C API for Tessera cluster data and rendering. This module lives in
 *  its own library (bf_tessera) so it can be developed and used independently
 *  from the core draw pipeline.
 */

#include <BLI_math_vector_types.hh>
#include <BLI_math_matrix_types.hh>
#include <cstdint>
#include <functional>
#include <vector>

// Forward declaration of Blender's Mesh type (defined in BKE_mesh.hh).
struct Mesh;

namespace blender {
namespace gpu {
class Texture;
}
namespace tessera {

using TesseraProgressCallback = std::function<void(float progress, const char *step_name)>;

struct TesseraMaterialTextures {
    gpu::Texture *diffuse = nullptr;
    gpu::Texture *normal = nullptr;
    gpu::Texture *roughness = nullptr;
    float4 base_color = float4(0.8f, 0.8f, 0.8f, 1.0f);
    float roughness_value = 0.5f;
    float metallic_value = 0.0f;
    bool has_diffuse = false;
    bool has_normal = false;
    bool has_roughness = false;
};

// ----------------------------------------------------------------------------
// Data model (matches Tessera_CUP2 .nc2 binary layout)
// ----------------------------------------------------------------------------

constexpr uint32_t kNc2Magic = 0x3255434E;  // "NCU2"
constexpr uint32_t kNc2Version = 1;

/** GPU-side vertex: quantized quantities expanded to float at runtime. */
struct TesseraVertex
{
    uint16_t px, py, pz;
    uint16_t nx, ny, nz;
    uint16_t tu, tv;
};

static_assert(sizeof(TesseraVertex) == 16, "TesseraVertex must be 16 bytes");

/**
 * LOD cluster (64 bytes, std430-compatible for direct GPU SSBO mapping).
 * Mirrors Tessera_CUP2 Node. The layout is shared with the GPU traversal
 * compute shader (#tessera_traverse.comp.glsl) — keep them in sync!
 */
struct TesseraCluster
{
    float sphere[4];        // own bounding sphere (xyz, r)
    float parentSphere[4];  // sphere used for parent error evaluation
    float error;            // own simplification error, world units
    float parentError;      // parent error (FLT_MAX for roots)
    uint32_t firstIndex;    // into index buffer
    uint32_t indexCount;
    int32_t  parentGroup;   // -1 for roots
    uint32_t group;         // own group id
    uint32_t level;         // 0 = finest
    uint16_t flags;         // bit0: has UV, bit1: has normals, bit2: has tangents
    uint16_t materialIndex; // material slot index
};

static_assert(sizeof(TesseraCluster) == 64, "TesseraCluster must be 64 bytes");

constexpr uint32_t kNc2FlagHasUV = (1u << 0);
constexpr uint32_t kNc2FlagHasNormals = (1u << 1);
constexpr uint32_t kNc2FlagHasTangents = (1u << 2);

/** Indirect draw command: matches VkDrawIndexedIndirectCommand. */
struct TesseraDrawCommand
{
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  vertexOffset;
    uint32_t firstInstance;
};

static_assert(sizeof(TesseraDrawCommand) == 20, "TesseraDrawCommand must be 20 bytes");

/** File header (matches Tessera_CUP2 Header). */
struct TesseraHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t clusterCount;
    uint32_t levelCount;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t proxyVertexCount;
    uint32_t proxyIndexCount;
    uint32_t maxClusterTriangles;
    uint32_t maxClusterVertices;
    uint32_t flags;  // bit0: has UV, bit1: has normals, bit2: has tangents
    uint32_t pageCount;  // v2 paged format: number of pages (0 for v1)
    float    boundsMin[3];
    float    boundsMax[3];
    float    uvMin[2];
    float    uvMax[2];
};

static_assert(sizeof(TesseraHeader) == 88, "TesseraHeader must be 88 bytes");

/** Top-level persistent Tessera data. */
struct TesseraData
{
    TesseraHeader header = {};
    std::vector<TesseraCluster> clusters;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<float4> tangents;
    std::vector<uint32_t> indices;
    std::vector<float3> proxyPositions;
    std::vector<uint32_t> proxyIndices;
};

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

/** Parameters controlling LOD selection. */
struct TesseraLodParams
{
    /** Screen-space error threshold in pixels. Smaller => more detail. */
    float pixelError = 1.0f;
    /** Maximum number of clusters to draw per frame. 0 = unlimited. */
    uint32_t maxClustersPerFrame = 0;
};

/**
 * Build Tessera cluster data from a Blender Mesh (drawn read-only).
 * The mesh is assumed to be in its evaluated/deformed state.
 * \returns true on success.
 */
bool buildFromMesh(const Mesh &mesh, TesseraData &out, TesseraProgressCallback progress_cb = nullptr);

/**
 * Serialize TesseraData into memory (matches Tessera_CUP2 .nc2 format).
 * \param data   source data
 * \param out    output buffer (caller owns)
 * \returns true on success
 */
bool serializeToMemory(const TesseraData &data, std::vector<uint8_t> &out);

/**
 * Deserialize TesseraData from memory produced by serializeToMemory.
 * \returns true on success
 */
bool deserializeFromMemory(TesseraData &out, const uint8_t *data, size_t size);

}  // namespace tessera
}  // namespace blender
