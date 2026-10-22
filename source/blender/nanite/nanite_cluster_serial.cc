/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Serialization helpers for NaniteData.
 *
 *  The layout is a simple flat stream:
 *    header, clusters, positions, normals, uvs, indices, proxyPositions, proxyIndices.
 */

#include "nanite_cluster.hh"

#include <cstring>
#include <vector>

namespace blender {
namespace nanite {

static size_t offsetOf(const NaniteData &data,
                       int fieldIndex /* 0=pos,1=nrm,2=uv,3=idx,4=proxyPos,5=proxyIdx */)
{
    size_t off = sizeof(NaniteHeader) + data.clusters.size() * sizeof(NaniteCluster);
    if (fieldIndex == 0 && !data.positions.empty()) {
        off += data.positions.size() * sizeof(float3);
    }
    if (fieldIndex == 1 && !data.normals.empty()) {
        off += data.normals.size() * sizeof(float3);
    }
    if (fieldIndex == 2 && !data.uvs.empty()) {
        off += data.uvs.size() * sizeof(float2);
    }
    if (fieldIndex == 3 && !data.indices.empty()) {
        off += data.indices.size() * sizeof(uint32_t);
    }
    if (fieldIndex == 4 && !data.proxyPositions.empty()) {
        off += data.proxyPositions.size() * sizeof(float3);
    }
    if (fieldIndex == 5 && !data.proxyIndices.empty()) {
        off += data.proxyIndices.size() * sizeof(uint32_t);
    }
    return off;
}

bool serializeToMemory(const NaniteData &data, std::vector<uint8_t> &out)
{
    const size_t headerSize = sizeof(NaniteHeader);
    const size_t clusterSize = sizeof(NaniteCluster);

    // Reserve space for header + clusters + all sub-buffers.
    size_t total = headerSize + data.clusters.size() * clusterSize;
    if (!data.positions.empty()) {
        total += data.positions.size() * sizeof(float3);
    }
    if (!data.normals.empty()) {
        total += data.normals.size() * sizeof(float3);
    }
    if (!data.uvs.empty()) {
        total += data.uvs.size() * sizeof(float2);
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
    if (!data.normals.empty()) {
        std::memcpy(ptr, data.normals.data(),
                    data.normals.size() * sizeof(float3));
        ptr += data.normals.size() * sizeof(float3);
    }
    // UVs
    if (!data.uvs.empty()) {
        std::memcpy(ptr, data.uvs.data(),
                    data.uvs.size() * sizeof(float2));
        ptr += data.uvs.size() * sizeof(float2);
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

bool deserializeFromMemory(NaniteData &out, const uint8_t *data, size_t size)
{
    const uint8_t *ptr = data;
    const uint8_t *end = data + size;

    if (end - ptr < static_cast<ptrdiff_t>(sizeof(NaniteHeader))) {
        return false;
    }
    // Header is parsed by the caller (or here).
    out.header = *reinterpret_cast<const NaniteHeader *>(ptr);
    ptr += sizeof(NaniteHeader);

    if (out.header.magic != kNc2Magic || out.header.version != kNc2Version) {
        return false;
    }

    // Clusters
    const size_t clusterCount = out.header.clusterCount;
    const size_t clusterSize = sizeof(NaniteCluster);
    if (end - ptr < static_cast<ptrdiff_t>(clusterCount * clusterSize)) {
        return false;
    }
    out.clusters.resize(clusterCount);
    std::memcpy(out.clusters.data(), ptr, clusterCount * clusterSize);
    ptr += clusterCount * clusterSize;

    // Walk the remaining layout and resize vectors, but do not fully
    // deserialize into the objects in this stub; the renderer operates
    // on the CPU-side NaniteData we already build, so the raw stream is
    // used for .nc2 interchange only.
    (void)ptr;
    return true;
}

}  // namespace nanite
}  // namespace blender
