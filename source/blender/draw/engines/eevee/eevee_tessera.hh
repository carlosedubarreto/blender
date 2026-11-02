/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * Integration of Tessera GPU cluster rendering into EEVEE.
 */

#pragma once

#include "tessera.h"
#include "tessera_draw.hh"

#include "BLI_map.hh"
#include "BLI_math_matrix_types.hh"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "DNA_object_types.h"
#include "DNA_scene_types.h"

namespace blender::draw {
class View;
class ObjectRef;
}

namespace blender::gpu {
class FrameBuffer;
}

namespace blender::tessera {
struct TesseraArchetype;
}

namespace blender::eevee {

bool DRW_eevee_tessera_instances_cached();
void DRW_eevee_tessera_instances_cache_invalidate();

class Instance;

class TesseraModule {
 private:
  Instance &inst_;

  struct ObjectState {
    std::string nc2_path;
    std::string data_source_key;
    int64_t file_mtime = -1;
    int64_t file_size = -1;
    bool used = false;
    uint32_t sync_round = 0;
    size_t last_instance_count = 0;
    int visualize_clusters = 0;
    bool tessera_enabled = true;
    float pixel_error = 1.0f;
    float culling_margin = 0.02f;
    int view_source = 0;
    float4x4 object_mat;
    Object *ob = nullptr;
    /* Name of the emitter object this state belongs to. */
    std::string ob_name;
    /* Archetype object name if resolved from a dupli / instanced collection. */
    std::string matched_archetype_name;
    /* Per-instance world matrices for scattered duplicates (Geometry Nodes /
     * ScatterFlow fields). Empty means a single draw with #object_mat. */
    std::vector<float4x4> instance_mats;
    /* Surviving instance matrices for the current frame after distance, screen-radius, and frustum culling */
    std::vector<float4x4> frame_visible_instances;
    /* Two-sided / foliage-card detection (drives opaque solid shading). */
    bool two_sided = false;
    /* True when this state is the base sync of a Geometry-Nodes instancer
     * whose evaluated geometry is purely an instance cloud (skip base draw). */
    bool is_instancer_base = false;
    bool show_stats = true;
    bool log_console = false;
    bool show_fps_panel = false;
    bool show_fps_hud = false;
    std::chrono::steady_clock::time_point last_frame_time{};
    std::chrono::steady_clock::time_point last_stat_time{};
    std::chrono::steady_clock::time_point last_dna_write_time{};
    bool was_navigating = false;
    float smoothed_dt = 0.0f;
    tessera::TesseraData data;
    std::vector<tessera::TesseraMaterialTextures> materials;
    std::vector<void *> cached_materials;
    int cached_totcol = -1;
    bool materials_extracted = false;
    bool traversed_this_frame = false;
    /* Archetype material base color (scattered foliage: emitter proxy has no
     * materials, the real plant does). */
    float4 base_color = float4(0.8f, 0.8f, 0.8f, 1.0f);
    bool base_color_valid = false;
    std::unique_ptr<tessera::TesseraDrawPass> pass;
    float current_tex_scale = 1.0f;
  };

  std::mutex mutex_;
  Map<std::string, std::unique_ptr<ObjectState>> objects_;

  uint32_t tessera_sync_round_ = 0;
  ObjectState *tessera_last_state_ = nullptr;
  const Object *tessera_last_ob_ = nullptr;
  const void *tessera_last_data_ = nullptr;
  uint32_t tessera_last_sync_round_ = 0;
  bool instances_cached_ = false;

  void update_telemetry(ObjectState &state, const float3 &cam_pos);

 public:
  TesseraModule(Instance &inst);
  ~TesseraModule();

  void begin_sync();
  void end_sync();
  bool instances_cached() const
  {
    return instances_cached_ && DRW_eevee_tessera_instances_cached();
  }
  bool has_valid_cache() const;
  void instances_cache_invalidate()
  {
    std::lock_guard lock(this->mutex_);
    instances_cached_ = false;
    DRW_eevee_tessera_instances_cache_invalidate();
  }
  bool is_enabled(const Scene *scene, Object *ob) const;
  /**
   * Record a Tessera-enabled object for rendering in the draw phase.
   * Handles single meshes, procedural instancers, and Geometry Nodes /
   * ScatterFlow dupli instance ranges.
   */
  bool record(const draw::ObjectRef &ob_ref);
  bool has_objects() const;
  bool has_cluster_visualization() const;
  void draw_prepass(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *depth_fb);
  void draw_gbuffer(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *gbuffer_fb);
  void draw_shadow(const draw::View &shadow_view, gpu::FrameBuffer *shadow_fb);
  void draw(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *target_fb);
};

bool DRW_eevee_tessera_instances_cached();
void DRW_eevee_tessera_instances_cache_invalidate();

}  // namespace blender::eevee
