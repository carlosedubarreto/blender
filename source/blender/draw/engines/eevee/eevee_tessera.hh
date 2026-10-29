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

namespace blender::draw {
class View;
}

namespace blender::gpu {
class FrameBuffer;
}

namespace blender::tessera {
struct TesseraArchetype;
}

namespace blender::eevee {

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
    int visualize_clusters = 0;
    bool tessera_enabled = true;
    float pixel_error = 1.0f;
    float culling_margin = 0.02f;
    int view_source = 0;
    float4x4 object_mat;
    Object *ob = nullptr;
    /* Name of the emitter object this state belongs to. */
    std::string ob_name;
    /* Per-instance world matrices for scattered duplicates (Geometry Nodes /
     * ScatterFlow fields). Empty means a single draw with #object_mat. */
    std::vector<float4x4> instance_mats;
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

  void update_telemetry(ObjectState &state, const float3 &cam_pos);

 public:
  TesseraModule(Instance &inst);
  ~TesseraModule();

  void begin_sync();
  bool is_enabled(Object *ob) const;
  /**
   * Record an object for Tessera rendering. For Geometry-Nodes / dupli
   * instance ranges the caller passes the full per-instance matrix array
   * (extracted from the ObjectRef instance range) so every scattered copy is
   * drawn and culled individually. \a object_mat is the base matrix of the
   * ref (the first instance matrix for range refs).
   */
  bool record(Object *ob,
              const float4x4 &object_mat,
              const float4x4 *instance_mats,
              int instance_count,
              const tessera::TesseraArchetype *archetype = nullptr,
              bool is_instancer_base = false);
  bool has_objects() const;
  bool has_cluster_visualization() const;
  void draw_prepass(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *depth_fb);
  void draw_gbuffer(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *gbuffer_fb);
  void draw_shadow(const draw::View &shadow_view, gpu::FrameBuffer *shadow_fb);
  void draw(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *target_fb);
};

}  // namespace blender::eevee
