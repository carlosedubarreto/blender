/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup nanite
 *
 *  Internal header for the Nanite cluster builder. Not part of the public API.
 */

#pragma once

#include "nanite.h"

#include <cfloat>
#include <limits>

namespace blender {
namespace nanite {

namespace detail {

// Quantize a float in [0,1] to a fixed-point u16.
inline uint16_t quantizeU16(float v)
{
    float clamped = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<uint16_t>(clamped * 65535.0f);
}

// Quantize a float in [-1,1] to a signed fixed-point u16 (for normals).
inline uint16_t quantizeS16(float v)
{
    float clamped = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<uint16_t>((clamped * 0.5f + 0.5f) * 65535.0f);
}

// Simple bounding sphere for a triangle.
void computeTriangleSphere(const float3 &a, const float3 &b, const float3 &c,
                           float3 &center, float &radius);

}  // namespace detail

bool buildFromMesh(const Mesh &mesh, NaniteData &out);

}  // namespace nanite
}  // namespace blender
