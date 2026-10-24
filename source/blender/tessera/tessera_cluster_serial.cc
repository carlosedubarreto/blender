/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Serialization helpers for TesseraData.
 *
 *  The layout is a simple flat stream:
 *    header, clusters, positions, normals, uvs, indices, proxyPositions, proxyIndices.
 */

#include "tessera_cluster.hh"

#include <cstring>
#include <vector>

namespace blender {
namespace tessera {

static size_t offsetOf(const TesseraData &data,
                       int fieldIndex /* 0=pos,1=nrm,2=uv,3=tan,4=idx,5=proxyPos,6=proxyIdx */)
{
    const bool has_normals = (data.header.flags & kNc2FlagHasNormals) != 0;
    const bool has_uvs = (data.header.flags & kNc2FlagHasUV) != 0;
    const bool has_tangents = (data.header.flags & kNc2FlagHasTangents) != 0;
    size_t off = sizeof(TesseraHeader) + data.clusters.size() * sizeof(TesseraCluster);
    if (fieldIndex <= 0) return off;
    if (!data.positions.empty()) off += data.positions.size() * sizeof(float3);
    if (fieldIndex <= 1) return off;
    if (has_normals && !data.normals.empty()) off += data.normals.size() * sizeof(float3);
    if (fieldIndex <= 2) return off;
    if (has_uvs && !data.uvs.empty()) off += data.uvs.size() * sizeof(float2);
    if (fieldIndex <= 3) return off;
    if (has_tangents && !data.tangents.empty()) off += data.tangents.size() * sizeof(float4);
    if (fieldIndex <= 4) return off;
    if (!data.indices.empty()) off += data.indices.size() * sizeof(uint32_t);
    if (fieldIndex <= 5) return off;
    if (!data.proxyPositions.empty()) off += data.proxyPositions.size() * sizeof(float3);
    return off;
}

bool serializeToMemory(const TesseraData &data, std::vector<uint8_t> &out)
{
    const size_t headerSize = sizeof(TesseraHeader);
    const size_t clusterSize = sizeof(TesseraCluster);
    const bool has_normals = (data.header.flags & kNc2FlagHasNormals) != 0;
    const bool has_uvs = (data.header.flags & kNc2FlagHasUV) != 0;
    const bool has_tangents = (data.header.flags & kNc2FlagHasTangents) != 0;

    // Reserve space for header + clusters + all sub-buffers.
    size_t total = headerSize + data.clusters.size() * clusterSize;
    if (!data.positions.empty()) {
        total += data.positions.size() * sizeof(float3);
    }
    if (has_normals && !data.normals.empty()) {
        total += data.normals.size() * sizeof(float3);
    }
    if (has_uvs && !data.uvs.empty()) {
        total += data.uvs.size() * sizeof(float2);
    }
    if (has_tangents && !data.tangents.empty()) {
        total += data.tangents.size() * sizeof(float4);
    }
    if (!data.indices.empty()) {
        total += data.indices.size() * sizeof(uint32_t);
    }
    if (!data.proxyPositions.empty()) {
        total += data.proxyPositions.size() * sizeof(float3);
    }
    if (!data.proxyIndices.empty()) {
        total += data.proxyIndices.size() * sizeof(uint32_t);
    }
    out.resize(total);

    uint8_t *ptr = out.data();

    // Header
    std::memcpy(ptr, &data.header, headerSize);
    ptr += headerSize;

    // Clusters
    std::memcpy(ptr, data.clusters.data(), data.clusters.size() * clusterSize);
    ptr += data.clusters.size() * clusterSize;

    // Positions
    if (!data.positions.empty()) {
        std::memcpy(ptr, data.positions.data(),
                    data.positions.size() * sizeof(float3));
        ptr += data.positions.size() * sizeof(float3);
    }
    // Normals
    if (has_normals && !data.normals.empty()) {
        std::memcpy(ptr, data.normals.data(),
                    data.normals.size() * sizeof(float3));
        ptr += data.normals.size() * sizeof(float3);
    }
    // UVs
    if (has_uvs && !data.uvs.empty()) {
        std::memcpy(ptr, data.uvs.data(),
                    data.uvs.size() * sizeof(float2));
        ptr += data.uvs.size() * sizeof(float2);
    }
    // Tangents
    if (has_tangents && !data.tangents.empty()) {
        std::memcpy(ptr, data.tangents.data(),
                    data.tangents.size() * sizeof(float4));
        ptr += data.tangents.size() * sizeof(float4);
    }

    // Indices
    if (!data.indices.empty()) {
        std::memcpy(ptr, data.indices.data(),
                    data.indices.size() * sizeof(uint32_t));
        ptr += data.indices.size() * sizeof(uint32_t);
    }

    // Proxy
    if (!data.proxyPositions.empty()) {
        std::memcpy(ptr, data.proxyPositions.data(),
                    data.proxyPositions.size() * sizeof(float3));
        ptr += data.proxyPositions.size() * sizeof(float3);
    }
    if (!data.proxyIndices.empty()) {
        std::memcpy(ptr, data.proxyIndices.data(),
                    data.proxyIndices.size() * sizeof(uint32_t));
        ptr += data.proxyIndices.size() * sizeof(uint32_t);
    }

    return true;
}

bool deserializeFromMemory(TesseraData &out, const uint8_t *data, size_t size)
{
    const uint8_t *ptr = data;
    const uint8_t *end = data + size;

    if (end - ptr < static_cast<ptrdiff_t>(sizeof(TesseraHeader))) {
        return false;
    }
    // Header is parsed by the caller (or here).
    out.header = *reinterpret_cast<const TesseraHeader *>(ptr);
    ptr += sizeof(TesseraHeader);

    if (out.header.magic != kNc2Magic || out.header.version != kNc2Version) {
        return false;
    }

    // Clusters
    const size_t clusterCount = out.header.clusterCount;
    const size_t clusterSize = sizeof(TesseraCluster);
    if (end - ptr < static_cast<ptrdiff_t>(clusterCount * clusterSize)) {
        return false;
    }
    out.clusters.resize(clusterCount);
    std::memcpy(out.clusters.data(), ptr, clusterCount * clusterSize);
    ptr += clusterCount * clusterSize;

    // Walk the remaining layout: positions, normals, uvs, indices,
    // proxyPositions, proxyIndices. A field is present when the writer had
    // data for it; presence is detected via the remaining stream size.
    auto read_field = [&](void *dst_ptr, size_t elem_size, size_t count) {
        const size_t bytes = count * elem_size;
        if (end - ptr < static_cast<ptrdiff_t>(bytes)) {
            return false;
        }
        if (count > 0) {
            std::memcpy(dst_ptr, ptr, bytes);
            ptr += bytes;
        }
        return true;
    };

    /* Positions: the vertex pool size is not stored in the header; recover it
     * from the remaining stream length by trying each field in order using
     * the counts implied by the cluster index ranges. The writer emits
     * positions (vertexCount), normals (same count when the normals flag is
     * set), uvs (same count), indices (indexCount), proxyPositions
     * (proxyVertexCount) and proxyIndices (proxyIndexCount). */
    const size_t pos_count = out.header.vertexCount;
    const size_t idx_count = out.header.indexCount;
    const size_t proxy_pos_count = out.header.proxyVertexCount;
    const size_t proxy_idx_count = out.header.proxyIndexCount;

    out.positions.resize(pos_count);
    if (!read_field(out.positions.data(), sizeof(float3), pos_count)) {
        return false;
    }
    /* Normals and UVs share the vertex-pool count when present; presence is
     * flagged in the header (bit0 = UV, bit1 = normals). */
    const bool has_normals = (out.header.flags & kNc2FlagHasNormals) != 0;
    const bool has_uvs = (out.header.flags & kNc2FlagHasUV) != 0;
    const bool has_tangents = (out.header.flags & kNc2FlagHasTangents) != 0;

    if (has_normals) {
        out.normals.resize(pos_count);
        if (!read_field(out.normals.data(), sizeof(float3), pos_count)) {
            return false;
        }
    }
    if (has_uvs) {
        out.uvs.resize(pos_count);
        if (!read_field(out.uvs.data(), sizeof(float2), pos_count)) {
            return false;
        }
    }
    if (has_tangents) {
        out.tangents.resize(pos_count);
        if (!read_field(out.tangents.data(), sizeof(float4), pos_count)) {
            return false;
        }
    }
    out.indices.resize(idx_count);
    if (!read_field(out.indices.data(), sizeof(uint32_t), idx_count)) {
        return false;
    }
    out.proxyPositions.resize(proxy_pos_count);
    if (!read_field(out.proxyPositions.data(), sizeof(float3),
                    proxy_pos_count))
    {
        return false;
    }
    out.proxyIndices.resize(proxy_idx_count);
    if (!read_field(out.proxyIndices.data(), sizeof(uint32_t),
                    proxy_idx_count))
    {
        return false;
    }
    return true;
}

}  // namespace tessera
}  // namespace blender
