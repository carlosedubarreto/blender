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

/** Instance buffer layout for multi-instance scatter / indirect draws. */
struct TesseraInstanceData {
    float model_matrix[16];
    float bounding_sphere[4]; /* xyz = center, w = radius */
    uint32_t object_id;
    uint32_t mesh_dag_root;
    uint32_t pad[2];
};

/**
 * Create the traversal compute shader. Returns null when compilation fails
 * or no GPU context is active; callers must fall back to the CPU traversal.
 * The result is owned by the caller (free with GPU_shader_free).
 */
gpu::Shader *createTraverseShader();

/**
 * Create the two-stage multi-instance traversal compute shader for massive
 * scatter / foliage rendering via indirect draws.
 */
gpu::Shader *createMultiInstanceTraverseShader();

}  // namespace tessera
}  // namespace blender