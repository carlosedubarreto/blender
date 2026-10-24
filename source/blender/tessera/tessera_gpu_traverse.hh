/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 *  \ingroup tessera
 *
 *  Phase 1: GPU-driven traversal.
 *
 *  A compute shader evaluates the (flat) LOD rule for every cluster in one
 *  dispatch: each thread independently frustum-culls its cluster, checks the
 *  emit rule (leaf, or parent error below the pixel threshold) and
 *  atomically appends an indirect draw command. No recursion, no queues, no
 *  per-frame CPU work beyond one dispatch and one 4-byte readback.
 */

#pragma once

#include "tessera_gpu_data.hh"

#include "GPU_shader.hh"

namespace blender {
namespace tessera {

/**
 * Create the traversal compute shader. Returns null when compilation fails
 * or no GPU context is active; callers must fall back to the CPU traversal.
 * The result is owned by the caller (free with GPU_shader_free).
 */
gpu::Shader *createTraverseShader();

}  // namespace tessera
}  // namespace blender