#pragma once

/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Public C API for Nanite cluster data and rendering. This module lives in
 *  its own library (bf_nanite) so it can be developed and used independently
 *  from the core draw pipeline.
 */

#include "BLI_math_base.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace blender {
namespace nanite {

// ----------------------------------------------------------------------------
// Data model (matches Nanite_CUP2 .nc2 binary layout)
// ----------------------------------------------------------------------------

constexpr uint32_t kNc2Magic = 0x3255434E;  // "NCU2"
constexpr uint32_t kNc2Version = 1;

/** GPU-side vertex: quantized quantities expanded to float at runtime. */
struct NaniteVertex
{
    uint16_t px, py, pz;
    uint16_t nx, ny, nz;
    uint16_t tu, tv;
};

static_assert(sizeof(NaniteVertex) == 16, "NaniteVertex must be 16 bytes");

/** LOD cluster (72 bytes). Mirrors Nanite_CUP2 Node. */
struct NaniteCluster
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
    uint32_t flags;
    uint32_t pad[2];
};

static_assert(sizeof(NaniteCluster) == 72, "NaniteCluster must be 72 bytes");

/** Indirect draw command: matches VkDrawIndexedIndirectCommand. */
struct NaniteDrawCommand
{
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  vertexOffset;
    uint32_t firstInstance;
};

static_assert(sizeof(NaniteDrawCommand) == 20, "NaniteDrawCommand must be 20 bytes");

/** File header (matches Nanite_CUP2 Header). */
struct NaniteHeader
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
    uint32_t flags;  // bit0: has UV, bit1: has normals
    uint32_t pad;
    float    boundsMin[3];
    float    boundsMax[3];
    float    uvMin[2];
    float    uvMax[2];
};

static_assert(sizeof(NaniteHeader) == 80, "NaniteHeader must be 80 bytes");

/** Top-level persistent Nanite data. */
struct NaniteData
{
    NaniteHeader header = {};
    std::vector<NaniteCluster> clusters;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint32_t> indices;
    std::vector<float3> proxyPositions;
    std::vector<uint32_t> proxyIndices;
};

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

/** Parameters controlling LOD selection. */
struct NaniteLodParams
{
    /** Screen-space error threshold in pixels. Smaller => more detail. */
    float pixelError = 1.0f;
    /** Maximum number of clusters to draw per frame. 0 = unlimited. */
    uint32_t maxClustersPerFrame = 0;
};

/**
 * Build Nanite cluster data from a Blender Mesh (drawn read-only).
 * The mesh is assumed to be in its evaluated/deformed state.
 * \returns true on success.
 */
bool buildFromMesh(const Mesh &mesh, NaniteData &out);

/**
 * Serialize NaniteData into memory (matches Nanite_CUP2 .nc2 format).
 * \param data   source data
 * \param out    output buffer (caller owns)
 * \returns true on success
 */
bool serializeToMemory(const NaniteData &data, std::vector<uint8_t> &out);

/**
 * Deserialize NaniteData from memory produced by serializeToMemory.
 * \returns true on success
 */
bool deserializeFromMemory(NaniteData &out, const uint8_t *data, size_t size);

}  // namespace nanite
}  // namespace blender
