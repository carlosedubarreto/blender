/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 *  \ingroup tessera
 *
 * Realized-geometry archetype resolution for Geometry-Nodes scatter fields.
 *
 * Modern Geometry-Nodes instancers (ScatterFlow etc.) realize their source
 * plant meshes and instance the realized geometry. The depsgraph delivers
 * those instances to the draw engines as duplis whose object is the *emitter*
 * (self-instance) but whose #DupliObject.ob_data is the realized source mesh.
 * A single emitter can scatter many different realized archetypes (one per
 * plant variety), each arriving as a separate ObjectRef instance range.
 *
 * This resolver maps a realized mesh back to its source-of-truth:
 *   1. Match the realized mesh (by vertex/corner counts) against the original
 *      mesh objects in #Main — the archetype object (e.g. "DewBerry_V1").
 *      When matched, the pre-built `<archetype>.nc2` cluster DAG next to the
 *      blend file is used (exact same geometry the native pipeline draws).
 *   2. Name / prefix fallback against original scene mesh archetypes matching
 *      the emitter / collection name (e.g. "DewBerry" matching "DewBerry_V1",
 *      "DewBerry_V2") when the dupli mesh is a low-poly proxy cube.
 *   3. When no original matches (joined / procedurally generated geometry),
 *      the cluster DAG is built directly from the realized mesh and cached
 *      for the session (keyed by the realized-mesh pointer).
 *
 * The result carries the archetype's material base color so engines can shade
 * the instances with the same color Workbench/EEVEE use natively.
 */

#pragma once

#include "tessera.h"

#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_sys_types.h"

#include <memory>
#include <mutex>
#include <string>

namespace blender {

struct ID;
struct Main;
struct Object;

namespace tessera {

/** Result of resolving one dupli range's realized geometry. */
struct TesseraArchetype {
  /** Stable per-geometry identifier ("DewBerry_V1" or "realized_5670v_37158c"). */
  std::string source_name;
  /** Absolute .nc2 path when a matching pre-built cluster file was found. */
  std::string nc2_path;
  int64_t nc2_mtime = -1;
  int64_t nc2_size = -1;
  /** Lazily built cluster data (only set when no .nc2 match was found). */
  std::shared_ptr<const TesseraData> built_data;
  /** Archetype material base color (Workbench material-color parity). */
  float4 base_color = float4(0.8f, 0.8f, 0.8f, 1.0f);
  bool base_color_valid = false;
  /** Archetype is an instanced object in #Main (engines may re-resolve it). */
  std::string matched_object_name;
  /** True when the geometry is likely alpha-cutout foliage cards. */
  bool two_sided = false;
  /** True for any successfully resolved realized-geometry archetype. */
  bool valid = false;
};

class ArchetypeResolver {
 public:
  static ArchetypeResolver &global();

  /**
   * Resolve the archetype geometry for one dupli range.
   * \param bmain: Main to search for original archetype objects (may be null).
   * \param blend_path: absolute path of the current blend file (for .nc2 lookup).
   * \param emitter: the instancer object the duplis belong to.
   * \param dupli_ob_data: the realized geometry the dupli range draws (or null).
   */
  TesseraArchetype resolve(const Main *bmain,
                           const char *blend_path,
                           const Object *emitter,
                           const ID *dupli_ob_data);

  /** Drop all lazily built data (e.g. after a file switch). */
  void clear();

 private:
  std::mutex mutex_;
  struct CacheEntry {
    /** Fully resolved archetype (matched .nc2 or lazily built data). */
    TesseraArchetype arch;
    /* Fingerprint guarding against pointer reuse by new geometry. */
    int64_t fp_verts = -1;
    int64_t fp_corners = -1;
  };
  /** Realized-mesh pointer -> resolved archetype. */
  Map<const void *, CacheEntry> resolved_;
};

}  // namespace tessera
}  // namespace blender
