/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_fileops.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix_types.hh"
#include "BLI_path_utils.hh"
#include "BLI_rect.h"
#include "BLI_string.h"

#include "DNA_fluid_types.h"
#include "DNA_ID.h"
#include "DNA_camera_types.h"

#include "BKE_camera.h"
#include "BKE_editmesh.hh"
#include "BKE_geometry_set.hh"
#include "BKE_idprop.hh"
#include "BKE_main.hh"
#include "BKE_material.hh"
#include "BKE_modifier.hh"
#include "BKE_object.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_particle.h"
#include "BKE_report.hh"

#include "DEG_depsgraph_query.hh"

#include "DNA_windowmanager_types.h"
#include "ED_paint.hh"
#include "ED_view3d.hh"

#include "BLT_translation.hh"

#include "GPU_context.hh"
#include "IMB_imbuf_types.hh"

#include "MEM_guardedalloc.h"

#include "RE_engine.h"
#include "RE_pipeline.h"

#include "draw_cache.hh"
#include "draw_common.hh"
#include "draw_sculpt.hh"
#include "draw_view_data.hh"

#include "tessera.h"
#include "tessera_archetype.hh"
#include "tessera_draw.hh"
#include "tessera_lod.hh"

#include "workbench_private.hh"

#include "workbench_engine.h" /* Own include. */

#include "BKE_idprop.hh"
#include "BKE_main.hh"
#include "WM_api.hh"
#include "WM_types.hh"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace blender {

static bool g_workbench_tessera_instances_cached = false;

bool DRW_workbench_tessera_instances_cached()
{
  return g_workbench_tessera_instances_cached;
}

namespace workbench {

using namespace draw;

class Instance : public DrawEngine {
 private:
  /* -------------------------------------------------------------------- */
  /** \name Tessera integration (experimental)
   *
   * Objects owning Tessera cluster data (custom IDProperty `tessera_nc2`)
   * are rendered through the Tessera visibility-buffer pipeline instead of
   * the regular Workbench mesh passes.
   * \{ */

  struct TesseraObjectState {
    std::string nc2_path;
    int64_t file_mtime = 0;
    int64_t file_size = -1;
    float4x4 object_mat = float4x4::identity();
    int visualize_clusters = 0;
    bool tessera_enabled = true;
    float pixel_error = 1.0f;
    float culling_margin = 0.02f;
    float lod_distance = 10.0f;
    float lod_intensity = 2.5f;
    /* LOD view source: 0 = viewport view, 1 = the scene's active camera. */
    int view_source = 0;
    bool used = false;
    Object *ob = nullptr;
    /* Name of the emitter object this state belongs to (differs from the map
     * key when a scatter field splits into per-archetype range states). */
    std::string ob_name;
    /* Resolved archetype object name (e.g. "SweetViolet_V1" from ScatterFlow) */
    std::string matched_archetype_name;
    /* Per-instance world matrices for scattered duplicates (Geometry Nodes /
     * ScatterFlow fields). Empty means a single draw with #object_mat. */
    std::vector<float4x4> instance_mats;
    /* Instances surviving this frame's camera-frustum cull (rebuilt in pass 1,
     * consumed by the batched indirect draws in pass 2). */
    std::vector<float4x4> frame_visible_instances;
    /* Distance-band sub-passes for scattered fields: instances at very
     * different camera distances need very different LOD cuts, so the
     * survivors are split into geometric distance bands and each band gets
     * its own traversal pass (evaluated at the band's nearest instance).
     * Band 0 reuses #pass; bands 1..N live here. */
    Vector<std::unique_ptr<blender::tessera::TesseraDrawPass>> band_passes;
    std::array<std::vector<float4x4>, 8> frame_band_instances;
    std::array<float, 8> frame_band_nearest_d = {0.0f};
    uint32_t frame_band_count = 0;
    /* Accelerated spatial BVH for hierarchical frustum culling of scattered instances. */
    tessera::InstanceBVH instance_bvh;
    /* Capacity watermark to prevent per-frame reallocations during sync. */
    size_t last_instance_count = 0;
    /* Sync round token: the instance list restarts when the base re-syncs. */
    uint32_t sync_round = 0;
    /* Two-sided / foliage-card detection (drives opaque solid shading). */
    bool two_sided = false;
    /* True for the base (non-dupli) sync of a Geometry-Nodes instancer: the
     * object's evaluated geometry is a pure instance cloud, so neither the
     * native pipeline nor Tessera draws anything for the base itself. */
    bool is_instancer_base = false;
    /* Archetype material base color (Workbench material-color parity for
     * scattered foliage whose emitter proxy carries no materials). */
    float4 base_color = float4(0.8f, 0.8f, 0.8f, 1.0f);
    bool base_color_valid = false;
    /* Identity of the geometry source feeding #data ("base" for the regular
     * .nc2 path, the archetype name for realized-geometry ranges). */
    std::string data_source_key;
    bool show_stats = true;
    bool log_console = false;
    bool show_fps_panel = false;
    bool show_fps_hud = false;
    std::chrono::steady_clock::time_point last_frame_time{};
    std::chrono::steady_clock::time_point last_dna_write_time{};
    bool was_navigating = false;
    float smoothed_dt = 0.0f;
    tessera::TesseraData data;
    std::unique_ptr<tessera::TesseraDrawPass> pass;
  };

  /** Tessera objects found during sync, keyed by "<emitter>|<geometry>". */
  Map<std::string, std::unique_ptr<TesseraObjectState>> tessera_objects_;
  /** Guards #tessera_objects_ (object_sync may run on extraction threads). */
  std::mutex tessera_mutex_;
  /** Bumped every begin_sync; instance lists restart when the base re-syncs. */
  uint32_t tessera_sync_round_ = 1;
  float4x4 tessera_last_view_persmat_ = float4x4::identity();
  bool tessera_has_last_view_persmat_ = false;
  std::chrono::steady_clock::time_point tessera_last_view_move_time_{};

  TesseraObjectState *tessera_last_state_ = nullptr;
  const Object *tessera_last_ob_ = nullptr;
  const void *tessera_last_data_ = nullptr;
  uint32_t tessera_last_sync_round_ = 0;


  /* Cap on collected per-object instance matrices (bounds per-frame cost of
   * scattered Geometry-Nodes / ScatterFlow fields). Up to 131k covers massive
   * point-cloud instancing output without truncating dense fields. */
  static constexpr size_t kMaxTesseraInstancesPerObject = 131072;

  /**
   * Detect two-sided / foliage-card geometry: explicit per-object IDProperty
   * override (`tessera_two_sided`) wins; without it the mesh is treated as
   * two-sided when its materials drive opacity from textures (the classic
   * alpha-cutout leaf-card setup). The visibility pipeline renders both
   * faces with flipped backface normals either way.
   */
  static bool tessera_object_is_two_sided(Object *ob)
  {
    if (ob == nullptr) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_two_sided");
    if (prop != nullptr && prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return false;
  }

  /* Tessera data ownership helpers. */

  static IDProperty *tessera_property_find_safe(Object *ob, const char *prop_name)
  {
    if (!ob) {
      return nullptr;
    }
    /* Check original datablock first (where user edits / Python properties live) */
    ID *orig_id = DEG_get_original_id(&ob->id);
    if (orig_id && orig_id->properties && orig_id->properties->type == IDP_GROUP) {
      IDProperty *prop = IDP_GetPropertyFromGroup(orig_id->properties, prop_name);
      if (prop) {
        return prop;
      }
    }
    if (ob->id.properties && ob->id.properties->type == IDP_GROUP) {
      return IDP_GetPropertyFromGroup(ob->id.properties, prop_name);
    }
    return nullptr;
  }

  static IDProperty *tessera_property_find(Object *ob, const char *prop_name)
  {
    return tessera_property_find_safe(ob, prop_name);
  }

  /**
   * Strict Tessera object qualification: procedural instancers and scatter
   * systems (e.g. ScatterFlow) generate helper empties, curve/pointcloud
   * objects and temporary evaluated objects. Those must never be registered
   * into or drawn through the Tessera pipeline. Visibility filtering is
   * already handled at sync time (DRW_object_is_renderable), so only the
   * hard structural requirements are enforced here.
   */
  static bool is_tessera_eligible_mesh(Object *ob)
  {
    if (ob == nullptr) {
      return false;
    }
    /* Strictly require mesh type and a valid mesh datablock. */
    if (ob->type != OB_MESH || ob->data == nullptr) {
      return false;
    }
    return true;
  }

  static bool tessera_path_get(Object *ob, std::string &r_path)
  {
    if (!ob) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_nc2");
    if (prop != nullptr) {
      if (prop->type == IDP_STRING && prop->data.pointer != nullptr) {
        r_path = static_cast<const char *>(prop->data.pointer);
        if (!r_path.empty()) {
          return true;
        }
      }
    }

    const char *blend_path = BKE_main_blendfile_path_from_global();
    if (blend_path != nullptr && blend_path[0] != '\0') {
      char test_path[FILE_MAX];
      BLI_snprintf(test_path, sizeof(test_path), "//%s.nc2", ob->id.name + 2);
      BLI_path_abs(test_path, blend_path);
      if (BLI_exists(test_path)) {
        r_path = test_path;
        return true;
      }
      if (ob->data != nullptr) {
        const ID *mesh_id = static_cast<const ID *>(ob->data);
        BLI_snprintf(test_path, sizeof(test_path), "//%s.nc2", mesh_id->name + 2);
        BLI_path_abs(test_path, blend_path);
        if (BLI_exists(test_path)) {
          r_path = test_path;
          return true;
        }
      }
    }
    return false;
  }

  static int tessera_visualize_get(Object *ob)
  {
    if (!ob) {
      return 0;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_visualize_clusters");
    if (prop == nullptr) {
      return 0;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop);
    }
    if (prop->type == IDP_FLOAT) {
      return int(IDP_float_get(prop));
    }
    if (prop->type == IDP_DOUBLE) {
      return int(IDP_double_get(prop));
    }
    return prop->data.val;
  }

  /** Master switch: false disables the Tessera path (falls back to the regular
   * mesh draw). Defaults to true when the property is absent. */
  static bool tessera_enabled_get(const Scene *scene, Object *ob)
  {
    if (!ob) {
      return false;
    }
    if (scene != nullptr) {
      if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "tessera_enabled")) {
        return (sprop->type == IDP_INT) ? (IDP_int_get(sprop) != 0) : (sprop->data.val != 0);
      }
      if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "scatterflow_anno_enabled")) {
        if ((sprop->type == IDP_INT && IDP_int_get(sprop) != 0) || sprop->data.val != 0) {
          return true;
        }
      }
      if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "scatterflow_nano")) {
        if ((sprop->type == IDP_INT && IDP_int_get(sprop) != 0) || sprop->data.val != 0) {
          return true;
        }
      }
      if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "use_scatterflow_nano")) {
        if ((sprop->type == IDP_INT && IDP_int_get(sprop) != 0) || sprop->data.val != 0) {
          return true;
        }
      }
    }
    /* If the object has a valid .nc2 asset path, activate Tessera automatically so
     * scenes loaded with baked tessera_enabled=0 still get full Tessera acceleration. */
    std::string path;
    if (tessera_path_get(ob, path) && !path.empty()) {
      return true;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_enabled");
    if (prop == nullptr) {
      return true;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return prop->data.val != 0;
  }

  static bool tessera_is_enabled(const Scene *scene, Object *ob)
  {
    if (ob == nullptr) {
      return false;
    }
    static thread_local const Object *s_last_ob = nullptr;
    static thread_local bool s_last_val = false;
    if (s_last_ob == ob) {
      return s_last_val;
    }
    s_last_ob = ob;
    const bool is_mesh = is_tessera_eligible_mesh(ob);
    const bool is_en = tessera_enabled_get(scene, ob);
    std::string path;
    const bool has_path = tessera_path_get(ob, path);

    if (!is_mesh || !is_en || !has_path) {
      s_last_val = false;
      return false;
    }
    s_last_val = true;
    return true;
  }

  /** LOD quality control: tolerated screen-space error in pixels. */
  static float tessera_pixel_error_get(Object *ob)
  {
    if (!ob) {
      return 1.0f;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_pixel_error");
    if (prop == nullptr) {
      return 1.0f;
    }
    float value = 1.0f;
    if (prop->type == IDP_DOUBLE) {
      value = float(IDP_double_get(prop));
    }
    else if (prop->type == IDP_FLOAT) {
      value = IDP_float_get(prop);
    }
    else if (prop->type == IDP_INT) {
      value = float(IDP_int_get(prop));
    }
    return (value > 0.0f) ? value : 1.0f;
  }

  static const IDProperty *tessera_scene_property_find_safe(const Scene *scene, const char *prop_name)
  {
    if (!scene) {
      return nullptr;
    }
    if (scene->id.properties && scene->id.properties->type == IDP_GROUP) {
      const IDProperty *prop = IDP_GetPropertyFromGroup(scene->id.properties, prop_name);
      if (prop) {
        return prop;
      }
    }
    ID *orig_id = DEG_get_original_id(const_cast<ID *>(&scene->id));
    if (orig_id && orig_id->properties && orig_id->properties->type == IDP_GROUP) {
      return IDP_GetPropertyFromGroup(orig_id->properties, prop_name);
    }
    return nullptr;
  }

  /**
   * Dedicated LOD threshold for scattered instance fields (Geometry Nodes /
   * ScatterFlow duplis). Scattered assets are individually tiny on screen
   * even when there are tens of thousands of them, so tolerating a coarser
   * cluster cut per instance trades a little per-instance detail for a large
   * reduction in rasterized cluster instances (the dominant cost of huge
   * fields). Defaults to 12 px; falls back to #tessera_pixel_error_get when
   * explicitly set to a non-positive value.
   */
  static float tessera_scatter_pixel_error_get(const Scene *scene, Object *ob, float base_pixel_error)
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_scatter_pixel_error");
      if (prop != nullptr) {
        float value = 12.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        /* Non-positive falls back to the regular per-object threshold. */
        if (value <= 0.0f) {
          return (base_pixel_error > 0.0f) ? base_pixel_error : 1.0f;
        }
        return value;
      }
    }

    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_scatter_pixel_error")) {
        float value = 12.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        if (value <= 0.0f) {
          return (base_pixel_error > 0.0f) ? base_pixel_error : 1.0f;
        }
        return value;
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_scatter_pixel_error")) {
        float value = 12.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        if (value <= 0.0f) {
          return (base_pixel_error > 0.0f) ? base_pixel_error : 2.0f;
        }
        return value;
      }
    }

    return 12.0f;
  }

  static float tessera_max_draw_distance_get(const Scene *scene,
                                             Object *ob,
                                             const std::string &ob_name,
                                             const std::string &arch_name = "")
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_max_draw_distance");
      if (prop != nullptr) {
        float value = 0.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        if (value > 0.0f) {
          return value;
        }
      }
    }

    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_max_draw_distance")) {
        float value = 0.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        if (value > 0.0f) {
          return value;
        }
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_max_draw_distance")) {
        float value = 0.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        if (value > 0.0f) {
          return value;
        }
      }
    }

    /* Asset-aware default draw distances:
     * Small details (Rocks_Small, Debris_Leafs, SweetViolet): cull beyond 20.0-22.0m
     * Medium foliage / shrubs (Galetta Grass, DewBerry, StickyWilly, StingingNettle): 28.0-35.0m
     * Large features (Rocks_Large, terrain): full view distance (0.0f) */
    const std::string name1 = ob_name.empty() && ob != nullptr ? std::string(ob->id.name + 2) : ob_name;
    const std::string name2 = arch_name;
    auto contains_ci = [](const std::string &haystack, const char *needle) {
      auto it = std::search(
          haystack.begin(), haystack.end(),
          needle, needle + strlen(needle),
          [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); }
      );
      return it != haystack.end();
    };
    auto contains_any = [&](const char *needle) {
      return contains_ci(name1, needle) || (!name2.empty() && contains_ci(name2, needle));
    };

    if (contains_any("Debris") || contains_any("LeafScatter")) {
      return 20.0f;
    }
    if (contains_any("Rocks_Small") || contains_any("SweetViolet") ||
        contains_any("Pebble") || contains_any("Stick"))
    {
      return 22.0f;
    }
    if (contains_any("Galetta") || contains_any("Grass")) {
      return 28.0f;
    }
    if (contains_any("StickyWilly") || contains_any("StingingNettle")) {
      return 30.0f;
    }
    if (contains_any("DewBerry") || contains_any("Shrub") ||
        contains_any("Bush") || contains_any("Flower") ||
        contains_any("Cactus"))
    {
      return 35.0f;
    }
    if (contains_any("Rocks_Large") || contains_any("Rock")) {
      return 0.0f;
    }

    return 0.0f;
  }

  static float tessera_culling_margin_get(const Scene *scene, Object *ob, Object *arch_ob = nullptr)
  {
    auto read_margin = [](Object *target) -> std::optional<float> {
      if (target == nullptr) {
        return std::nullopt;
      }
      if (const IDProperty *prop = tessera_property_find_safe(target, "tessera_culling_margin")) {
        if (prop->type == IDP_DOUBLE) {
          return float(IDP_double_get(prop));
        }
        if (prop->type == IDP_FLOAT) {
          return IDP_float_get(prop);
        }
        if (prop->type == IDP_INT) {
          return float(IDP_int_get(prop));
        }
      }
      if (const IDProperty *prop = tessera_property_find_safe(target, "culling_margin")) {
        if (prop->type == IDP_DOUBLE) {
          return float(IDP_double_get(prop));
        }
        if (prop->type == IDP_FLOAT) {
          return IDP_float_get(prop);
        }
        if (prop->type == IDP_INT) {
          return float(IDP_int_get(prop));
        }
      }
      return std::nullopt;
    };

    if (auto m = read_margin(ob)) {
      return std::clamp(*m, -1000.0f, 1000.0f);
    }
    if (auto m = read_margin(arch_ob)) {
      return std::clamp(*m, -1000.0f, 1000.0f);
    }

    if (scene != nullptr) {
      auto read_scene_margin = [](const Scene *sc, const char *prop_name) -> std::optional<float> {
        if (const IDProperty *prop = tessera_scene_property_find_safe(sc, prop_name)) {
          if (prop->type == IDP_DOUBLE) {
            return float(IDP_double_get(prop));
          }
          if (prop->type == IDP_FLOAT) {
            return IDP_float_get(prop);
          }
          if (prop->type == IDP_INT) {
            return float(IDP_int_get(prop));
          }
        }
        return std::nullopt;
      };
      if (auto m = read_scene_margin(scene, "tessera_batch_culling_margin")) {
        return std::clamp(*m, -1000.0f, 1000.0f);
      }
      if (auto m = read_scene_margin(scene, "batch_culling_margin")) {
        return std::clamp(*m, -1000.0f, 1000.0f);
      }
      if (auto m = read_scene_margin(scene, "tessera_culling_margin")) {
        return std::clamp(*m, -1000.0f, 1000.0f);
      }
      if (auto m = read_scene_margin(scene, "culling_margin")) {
        return std::clamp(*m, -1000.0f, 1000.0f);
      }
    }

    return 0.02f;
  }

  static float tessera_lod_distance_get(const Scene *scene, Object *ob)
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_lod_distance");
      if (prop != nullptr) {
        float value = 10.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.1f) ? value : 0.1f;
      }
    }

    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_lod_distance")) {
        float value = 10.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.1f) ? value : 0.1f;
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_lod_distance")) {
        float value = 10.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.1f) ? value : 0.1f;
      }
    }

    return 10.0f;
  }

  static float tessera_lod_intensity_get(const Scene *scene, Object *ob)
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_lod_intensity");
      if (prop != nullptr) {
        float value = 2.5f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
    }

    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_lod_intensity")) {
        float value = 2.5f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_lod_intensity")) {
        float value = 2.5f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
    }

    return 2.5f;
  }

  static float tessera_min_screen_radius_get(const Scene *scene, Object *ob, const std::string &arch_name = "")
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_min_screen_radius");
      if (prop != nullptr) {
        float value = 3.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
    }
    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_min_screen_radius")) {
        float value = 3.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_min_screen_radius")) {
        float value = 3.0f;
        if (prop->type == IDP_DOUBLE) {
          value = float(IDP_double_get(prop));
        }
        else if (prop->type == IDP_FLOAT) {
          value = IDP_float_get(prop);
        }
        else if (prop->type == IDP_INT) {
          value = float(IDP_int_get(prop));
        }
        return (value >= 0.0f) ? value : 0.0f;
      }
    }
    const std::string name1 = (ob != nullptr ? std::string(ob->id.name + 2) : "");
    const std::string name2 = arch_name;
    auto contains_ci = [](const std::string &haystack, const char *needle) {
      auto it = std::search(
          haystack.begin(), haystack.end(),
          needle, needle + strlen(needle),
          [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); }
      );
      return it != haystack.end();
    };
    auto contains_any = [&](const char *needle) {
      return (!name1.empty() && contains_ci(name1, needle)) ||
             (!name2.empty() && contains_ci(name2, needle));
    };

    /* Asset-calibrated sub-pixel thresholds for scatter fields:
     * Tiny clutter (Rocks_Small, SweetViolet, Debris, Sticks, Pebbles) are tiny (2-8cm);
     * culling them below 4.5px removes tens of thousands of sub-pixel instances with zero visual degradation.
     * Medium foliage (Galetta Grass, DewBerry, StickyWilly, StingingNettle): 2.5px.
     * Large rocks and trees: 1.0px. */
    if (contains_any("Rocks_Small") || contains_any("SweetViolet") ||
        contains_any("Debris") || contains_any("LeafScatter") ||
        contains_any("Stick") || contains_any("Pebble"))
    {
      return 4.5f;
    }
    if (contains_any("Galetta") || contains_any("DewBerry") ||
        contains_any("StickyWilly") || contains_any("StingingNettle") ||
        contains_any("Grass") || contains_any("Bush") || contains_any("Flower"))
    {
      return 2.5f;
    }
    return 1.5f;
  }

  /** LOD view source: 0 = viewport view, 1 = the scene's active camera. */
  static int tessera_view_source_get(Object *ob, Object *arch_ob = nullptr)
  {
    auto read_vs = [](Object *target) -> std::optional<int> {
      if (!target) {
        return std::nullopt;
      }
      const IDProperty *prop = tessera_property_find_safe(target, "tessera_view_source");
      if (prop != nullptr && prop->type == IDP_INT) {
        return IDP_int_get(prop);
      }
      return std::nullopt;
    };
    if (auto vs = read_vs(arch_ob)) {
      return *vs;
    }
    if (auto vs = read_vs(ob)) {
      return *vs;
    }
    return 0; /* Safe default (Viewport Mode). */
  }

  static bool tessera_show_stats_get(Object *ob)
  {
    if (!ob) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_show_stats");
    if (prop == nullptr) {
      return false;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return prop->data.val != 0;
  }

  static bool tessera_log_console_get(Object *ob)
  {
    if (!ob) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_log_console");
    if (prop == nullptr) {
      return false;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return prop->data.val != 0;
  }

  static bool tessera_show_fps_panel_get(Object *ob)
  {
    if (!ob) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_show_fps_panel");
    if (prop == nullptr) {
      return false;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return prop->data.val != 0;
  }

  static bool tessera_show_fps_hud_get(Object *ob)
  {
    if (!ob) {
      return false;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_show_fps_hud");
    if (prop == nullptr) {
      return false;
    }
    if (prop->type == IDP_INT) {
      return IDP_int_get(prop) != 0;
    }
    return prop->data.val != 0;
  }

  static void idp_int_replace(IDProperty *idgroup, const char *name, int value)
  {
    if (idgroup == nullptr || idgroup->type != IDP_GROUP) {
      return;
    }
    if (IDProperty *prev = IDP_GetPropertyFromGroup(idgroup, name)) {
      if (ELEM(prev->type, IDP_INT, IDP_BOOLEAN)) {
        IDP_int_or_bool_set(prev, value);
        return;
      }
      IDP_FreeFromGroup(idgroup, prev);
    }
    IDP_AddToGroup(idgroup, bke::idprop::create(name, value).release());
  }

  static void idp_float_replace(IDProperty *idgroup, const char *name, float value)
  {
    if (idgroup == nullptr || idgroup->type != IDP_GROUP) {
      return;
    }
    if (IDProperty *prev = IDP_GetPropertyFromGroup(idgroup, name)) {
      if (prev->type == IDP_FLOAT) {
        IDP_float_set(prev, value);
        return;
      }
      if (prev->type == IDP_DOUBLE) {
        IDP_double_set(prev, double(value));
        return;
      }
      IDP_FreeFromGroup(idgroup, prev);
    }
    IDP_AddToGroup(idgroup, bke::idprop::create(name, value).release());
  }

  static void tessera_write_fps_to_object(Object *ob, float fps, float frame_time_ms)
  {
    if (ob == nullptr) {
      return;
    }
    auto write_props = [&](ID *id) {
      if (id == nullptr) {
        return;
      }
      if (IDProperty *group = IDP_EnsureProperties(id)) {
        idp_float_replace(group, "tessera_stat_fps", fps);
        idp_float_replace(group, "tessera_stat_frametime", frame_time_ms);
      }
    };

    /* Only write to evaluated datablock during rendering; modifying the original
     * datablock tags the dependency graph as dirty and forces modifier re-evaluation. */
    write_props(&ob->id);
  }

  static void tessera_write_telemetry_to_object(Object *ob,
                                                const tessera::TesseraTelemetry &telem,
                                                const tessera::TesseraMemoryBudget *budget = nullptr,
                                                uint32_t total_instances = 0,
                                                uint32_t active_instances = 0)
  {
    if (ob == nullptr) {
      return;
    }
    auto write_props = [&](ID *id) {
      if (id == nullptr) {
        return;
      }
      if (IDProperty *group = IDP_EnsureProperties(id)) {
        idp_int_replace(group, "tessera_stat_clusters", int(telem.active_clusters));
        idp_int_replace(group, "tessera_stat_triangles", int(telem.active_triangles));
        idp_int_replace(group, "tessera_stat_lod_min", int(telem.active_lod_min));
        idp_int_replace(group, "tessera_stat_lod_max", int(telem.active_lod_max));
        idp_int_replace(group, "tessera_total_clusters", int(telem.total_clusters));
        idp_int_replace(group, "tessera_total_triangles", int(telem.total_triangles));
        idp_float_replace(group, "tessera_stat_distance", telem.cam_distance);
        idp_float_replace(group, "tessera_stat_pixel_error", telem.pixel_error);
        idp_float_replace(group, "tessera_stat_error", telem.pixel_error);
        idp_float_replace(group, "tessera_stat_culling_margin", telem.culling_margin);
        idp_float_replace(group, "tessera_stat_tex_scale", telem.tex_scale);
        idp_int_replace(group, "tessera_stat_tex_res_x", telem.tex_res_x);
        idp_int_replace(group, "tessera_stat_tex_res_y", telem.tex_res_y);
        idp_float_replace(group, "tessera_stat_dispatch_ms", telem.dispatch_time_ms);
        idp_float_replace(group, "tessera_stat_gpu_traverse_ms", telem.gpu_traverse_ms);
        idp_float_replace(group, "tessera_stat_gpu_raster_ms", telem.gpu_raster_ms);
        idp_float_replace(group, "tessera_stat_gpu_total_ms", telem.gpu_total_ms);
        idp_int_replace(group, "tessera_stat_sync_stalls", int(telem.sync_stalls));
        if (budget != nullptr) {
          idp_float_replace(group, "tessera_stat_vram_resident_mb", float(budget->current_resident_bytes) / (1024.0f * 1024.0f));
          idp_float_replace(group, "tessera_stat_vram_budget_mb", float(budget->max_vram_bytes) / (1024.0f * 1024.0f));
          idp_float_replace(group, "tessera_stat_disk_read_mbs", budget->disk_read_mbs);
          idp_float_replace(group, "tessera_stat_cache_hit_ratio", budget->cache_hit_ratio);
          idp_int_replace(group, "tessera_stat_total_instances", int(total_instances));
          idp_int_replace(group, "tessera_stat_active_instances", int(active_instances));
          idp_int_replace(group, "tessera_stat_pending_requests", int(budget->pending_fetch_bytes));
        }
      }
    };

    /* Only write to evaluated datablock during rendering; modifying the original
     * datablock tags the dependency graph as dirty and forces modifier re-evaluation. */
    write_props(&ob->id);
  }

  static void tessera_stat_culling_margin_write(Object *ob, float margin)
  {
    if (ob == nullptr) {
      return;
    }
    auto write_props = [&](ID *id) {
      if (id == nullptr) {
        return;
      }
      if (IDProperty *group = IDP_EnsureProperties(id)) {
        idp_float_replace(group, "tessera_stat_culling_margin", margin);
      }
    };

    /* Only write to evaluated datablock during rendering; modifying the original
     * datablock tags the dependency graph as dirty and forces modifier re-evaluation. */
    write_props(&ob->id);
  }

  /** Write an integer statistic to the object and its original datablock. */
  static void tessera_stat_int_write(Object *ob, const char *prop_name, int value)
  {
    if (ob == nullptr) {
      return;
    }
    auto write_props = [&](ID *id) {
      if (id == nullptr) {
        return;
      }
      if (IDProperty *group = IDP_EnsureProperties(id)) {
        idp_int_replace(group, prop_name, value);
      }
    };

    /* Only write to evaluated datablock during rendering; modifying the original
     * datablock tags the dependency graph as dirty and forces modifier re-evaluation. */
    write_props(&ob->id);
  }

  /**
   * Re-resolve the evaluated object for a recorded Tessera object by name.
   * The pointer stored during sync may have been freed by the depsgraph in
   * between frames (procedural instancers and scatter systems recreate their
   * evaluated objects every frame), so it must never be dereferenced at draw
   * time. Returns nullptr when the object no longer exists.
   */
  static Object *tessera_object_revalidate(const DRWContext *draw_ctx, const std::string &name)
  {
    if (draw_ctx == nullptr || draw_ctx->depsgraph == nullptr) {
      return nullptr;
    }
    Main *bmain = DEG_get_bmain(draw_ctx->depsgraph);
    if (bmain == nullptr) {
      return nullptr;
    }
    Object *orig = static_cast<Object *>(
        BLI_findstring(&bmain->objects, name.c_str(), offsetof(ID, name) + 2));
    if (orig == nullptr) {
      return nullptr;
    }
    return DEG_get_evaluated(draw_ctx->depsgraph, orig);
  }

  /**
   * Projection*view matrix of the scene's active camera, fitted to the target
   * resolution. Returns false when the scene has no usable camera.
   */
  static bool tessera_camera_persmat(const Scene *scene,
                                    const DRWContext *draw_ctx,
                                    const int2 &resolution,
                                    const Object *state_ob,
                                    float4x4 &r_persmat,
                                    float &r_proj_scale_y,
                                    const Object *arch_ob = nullptr)
  {
    if (scene == nullptr || draw_ctx->depsgraph == nullptr) {
      return false;
    }
    const Object *cam_ob = nullptr;
    auto find_cull_cam = [&](const Object *target) -> const Object * {
      if (target == nullptr) {
        return nullptr;
      }
      IDProperty *cull_cam_prop = tessera_property_find_safe(const_cast<Object *>(target), "tessera_cull_camera");
      if (cull_cam_prop != nullptr && cull_cam_prop->type == IDP_STRING) {
        const char *cam_name = IDP_string_get(cull_cam_prop);
        if (cam_name != nullptr && cam_name[0] != '\0') {
          Main *bmain = DEG_get_bmain(draw_ctx->depsgraph);
          if (bmain != nullptr) {
            Object *found = static_cast<Object *>(
                BLI_findstring(&bmain->objects, cam_name, offsetof(ID, name) + 2));
            if (found != nullptr) {
              return DEG_get_evaluated(draw_ctx->depsgraph, found);
            }
          }
        }
      }
      return nullptr;
    };

    if (arch_ob != nullptr) {
      cam_ob = find_cull_cam(arch_ob);
    }
    if (cam_ob == nullptr && state_ob != nullptr) {
      cam_ob = find_cull_cam(state_ob);
    }
    if (cam_ob == nullptr && scene->camera != nullptr) {
      cam_ob = DEG_get_evaluated(draw_ctx->depsgraph, scene->camera);
    }
    if (cam_ob == nullptr || cam_ob->type != OB_CAMERA || cam_ob->data == nullptr) {
      return false;
    }
    CameraParams params;
    BKE_camera_params_init(&params);
    BKE_camera_params_from_object(&params, cam_ob);
    const int winx = (scene->r.xsch * scene->r.size) / 100;
    const int winy = (scene->r.ysch * scene->r.size) / 100;
    BKE_camera_params_compute_viewplane(
        &params, winx > 0 ? winx : resolution.x, winy > 0 ? winy : resolution.y, scene->r.xasp, scene->r.yasp);
    BKE_camera_params_compute_matrix(&params);
    float4x4 winmat = float4x4::identity();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        winmat[i][j] = params.winmat[i][j];
      }
    }
    const float4x4 viewmat = math::invert(cam_ob->object_to_world());
    r_persmat = winmat * viewmat;
    r_proj_scale_y = std::abs(winmat[1][1]);
    return true;
  }

  /**
   * Record a Tessera-enabled object for rendering in the draw phase.
   * \return true when the object is owned by the Tessera pipeline (the caller
   * must skip the regular Workbench mesh draw for it).
   */
  bool tessera_record(ObjectRef &ob_ref)
  {
    Object *ob = ob_ref.object;
    if (ob == nullptr) {
      return false;
    }

    const bool is_dupli = (ob->base_flag & BASE_FROM_DUPLI) != 0;
    const bool is_range = ob_ref.is_range();

    /* Fast path for consecutive instances of the same object / archetype in scatter fields.
     * Bypasses IDProperty lookups, path conversions, archetype fingerprinting and hash lookups. */
    if (tessera_last_state_ != nullptr &&
        tessera_last_ob_ == ob &&
        tessera_last_data_ == ob->data &&
        tessera_last_sync_round_ == this->tessera_sync_round_)
    {
      if (is_range) {
        const int range_count = ob_ref.instances_count();
        const size_t room = (tessera_last_state_->instance_mats.size() < kMaxTesseraInstancesPerObject) ?
                                (kMaxTesseraInstancesPerObject - tessera_last_state_->instance_mats.size()) :
                                0;
        const int to_copy = std::min(int(room), range_count);
        const size_t old_sz = tessera_last_state_->instance_mats.size();
        tessera_last_state_->instance_mats.resize(old_sz + to_copy);
        for (int i = 0; i < to_copy; ++i) {
          tessera_last_state_->instance_mats[old_sz + i] = ob_ref.object_to_world(i);
        }
      }
      else if (is_dupli) {
        if (tessera_last_state_->instance_mats.size() < kMaxTesseraInstancesPerObject) {
          tessera_last_state_->instance_mats.push_back(ob->object_to_world());
        }
      }
      return true;
    }

    if (!is_tessera_eligible_mesh(ob)) {
      return false;
    }
    if (!tessera_enabled_get(scene_state_.scene, ob)) {
      return false;
    }
    std::string idprop_path;
    if (!tessera_path_get(ob, idprop_path)) {
      return false;
    }

    /* Resolve blend-relative paths. */
    char abs_path[FILE_MAX];
    BLI_strncpy(abs_path, idprop_path.c_str(), sizeof(abs_path));
    BLI_path_abs(abs_path, BKE_main_blendfile_path_from_global());

    /* Realized-geometry archetype resolution: Geometry-Nodes scatter fields
     * (ScatterFlow etc.) deliver their instanced plant meshes as duplis of
     * the emitter whose #DupliObject.ob_data is the *realized source mesh*,
     * one ObjectRef instance range per plant variety. Resolve each range's
     * real geometry (the pre-built archetype .nc2, or a lazily built cluster
     * DAG) so Tessera draws the same plants the native pipeline renders
     * instead of the low-poly emitter proxy. */
    tessera::TesseraArchetype arch;
    bool arch_resolved = false;
    Main *arch_bmain = (this->draw_ctx != nullptr && this->draw_ctx->depsgraph != nullptr) ?
                           DEG_get_bmain(this->draw_ctx->depsgraph) :
                           nullptr;
    if (is_dupli || is_range) {
      if (const ID *range_data = static_cast<const ID *>(ob->data)) {
        arch = tessera::ArchetypeResolver::global().resolve(
            arch_bmain, BKE_main_blendfile_path_from_global(), ob, range_data);
        arch_resolved = arch.valid;
      }
    }

    std::string state_key(ob->id.name + 2);
    if (arch_resolved) {
      state_key += "|";
      state_key += arch.source_name;
    }
    std::unique_ptr<TesseraObjectState> &entry = tessera_objects_.lookup_or_add_default(
        state_key);
    if (entry == nullptr) {
      entry = std::make_unique<TesseraObjectState>();
    }
    TesseraObjectState &state = *entry;

    tessera_last_state_ = &state;
    tessera_last_ob_ = ob;
    tessera_last_data_ = ob->data;
    tessera_last_sync_round_ = this->tessera_sync_round_;

    if (state.sync_round != this->tessera_sync_round_) {
      /* First sync of this object in the current sync round: start a fresh
       * instance list and preserve vector capacity to avoid malloc churn. */
      state.sync_round = this->tessera_sync_round_;
      state.last_instance_count = state.instance_mats.size();
      state.instance_mats.clear();
      if (state.last_instance_count > 0) {
        state.instance_mats.reserve(state.last_instance_count);
      }

      state.used = true;
      state.ob = ob;
      state.ob_name = std::string(ob->id.name + 2);
      state.matched_archetype_name.clear();
      if (arch_resolved && !arch.matched_object_name.empty()) {
        state.matched_archetype_name = arch.matched_object_name;
      }
      Object *arch_ob = (!state.matched_archetype_name.empty() && arch_bmain != nullptr) ?
                            static_cast<Object *>(BLI_findstring(&arch_bmain->objects,
                                                                 state.matched_archetype_name.c_str(),
                                                                 offsetof(ID, name) + 2)) :
                            nullptr;
      state.show_stats = tessera_show_stats_get(ob);
      state.log_console = tessera_log_console_get(ob);
      state.show_fps_panel = tessera_show_fps_panel_get(ob);
      state.show_fps_hud = tessera_show_fps_hud_get(ob);
      state.visualize_clusters = tessera_visualize_get(ob);
      state.pixel_error = tessera_pixel_error_get(ob);
      state.view_source = tessera_view_source_get(ob, arch_ob);
      state.culling_margin = tessera_culling_margin_get(scene_state_.scene, ob, arch_ob);
      state.lod_distance = tessera_lod_distance_get(scene_state_.scene, ob);
      state.lod_intensity = tessera_lod_intensity_get(scene_state_.scene, ob);
      state.object_mat = ob_ref.is_range() ? ob_ref.object_to_world(0) : ob->object_to_world();
      state.two_sided = tessera_object_is_two_sided(ob);
      state.is_instancer_base = false;
      if (arch_resolved) {
        /* Shading parity with the native pipeline: the archetype's material
         * color and two-sided foliage-card handling apply to every instance
         * of this range (the emitter proxy itself carries no materials). */
        if (arch.base_color_valid) {
          state.base_color = arch.base_color;
          state.base_color_valid = true;
        }
        if (arch_ob != nullptr) {
          state.two_sided = state.two_sided || tessera_object_is_two_sided(arch_ob);
        }
        state.two_sided = state.two_sided || arch.two_sided;
      }
    }
    if (!is_dupli && !is_range) {
      /* Base sync of a pure Geometry-Nodes or dupli instancer: the evaluated geometry
       * is an instance cloud, so the native pipeline draws nothing for the
       * base object itself — skip its Tessera draw to stay identical. */
      const bool is_instancer = (ob->transflag & OB_DUPLI) != 0 ||
                                (ob->runtime && ob->runtime->geometry_set_eval &&
                                 ob->runtime->geometry_set_eval->has_instances());
      if (is_instancer) {
        state.is_instancer_base = true;
        /* Preserve all child archetype states cached for this instancer across frames */
        const std::string prefix = std::string(ob->id.name + 2) + "|";
        for (auto item : tessera_objects_.items()) {
          if (item.key.rfind(prefix, 0) == 0) {
            item.value->used = true;
          }
        }
      }
    }
    /* ScatterFlow / Geometry-Nodes instances arrive as ObjectRef instance
     * ranges: extract the full 4x4 matrix of every instance dynamically into
     * the instance array (up to 64k entries). Each scattered copy is culled
     * and rasterized individually at draw time — this is what restores the
     * scattered foliage field that previously collapsed to a single card. */
    if (ob_ref.is_range()) {
      const int range_count = ob_ref.instances_count();
      const size_t room = (state.instance_mats.size() < kMaxTesseraInstancesPerObject) ?
                              (kMaxTesseraInstancesPerObject - state.instance_mats.size()) :
                              0;
      const int to_copy = std::min(int(room), range_count);
      const size_t old_sz = state.instance_mats.size();
      state.instance_mats.resize(old_sz + to_copy);
      for (int i = 0; i < to_copy; ++i) {
        state.instance_mats[old_sz + i] = ob_ref.object_to_world(i);
      }
    }
    else if (is_dupli) {
      /* Legacy per-dupli sync path (one object_sync call per dupli). */
      if (state.instance_mats.size() < kMaxTesseraInstancesPerObject) {
        state.instance_mats.push_back(state.object_mat);
      }
    }

    /* (Re)load the cluster data when the resolved source changed. */
    if (arch_resolved && arch.built_data != nullptr) {
      /* Clusters were built directly from the realized geometry: adopt the
       * cached data (reload only when the geometry identity changed). */
      if (state.data_source_key != arch.source_name || state.data.clusters.empty()) {
        state.data = *arch.built_data;
        state.pass.reset();
        state.band_passes.clear();
      }
      state.data_source_key = arch.source_name;
      state.nc2_path = "";
      state.file_mtime = 0;
      state.file_size = 0;
    }
    else {
      const std::string load_path = arch_resolved ? arch.nc2_path : std::string(abs_path);
      const bool in_interactive_viewport = (this->draw_ctx != nullptr &&
                                            this->draw_ctx->rv3d != nullptr);
      bool stale = (state.nc2_path != load_path) || state.data.clusters.empty();
      int64_t load_mtime = state.file_mtime;
      int64_t load_size = state.file_size;
      if (stale || !in_interactive_viewport) {
        BLI_stat_t load_info;
        if (BLI_stat(load_path.c_str(), &load_info) == 0) {
          load_mtime = int64_t(load_info.st_mtime);
          load_size = int64_t(load_info.st_size);
        }
        else {
          load_mtime = -1;
          load_size = -1;
        }
        stale = stale || (state.file_mtime != load_mtime) || (state.file_size != load_size);
      }
      if (stale || state.data.clusters.empty()) {
        size_t data_size = 0;
        if (void *file_data = BLI_file_read_binary_as_mem(load_path.c_str(), 0, &data_size)) {
          tessera::TesseraData loaded;
          if (tessera::deserializeFromMemory(loaded,
                                            static_cast<const uint8_t *>(file_data),
                                            data_size) &&
              !loaded.clusters.empty())
          {
            state.data = std::move(loaded);
          }
          MEM_delete_void(file_data);
        }
        state.file_mtime = load_mtime;
        state.file_size = load_size;
        state.nc2_path = load_path;
        state.data_source_key = arch_resolved ? arch.source_name : "base";
        /* Drop the pass so it is rebuilt with the new data on draw. */
        state.pass.reset();
        state.band_passes.clear();
      }
    }
    /* Objects whose data failed to load fall back to the regular mesh path. */
    return !state.data.clusters.empty();
  }

  /** Render all Tessera objects recorded during sync into the target. */
  void tessera_draw(const int2 &resolution)
  {
    if (tessera_objects_.is_empty()) {
      return;
    }
    std::lock_guard lock(this->tessera_mutex_);
    const draw::View &default_view = draw::View::default_get();
    const float4x4 unjittered_persmat = default_view.persmat();
    const float4x4 viewport_persmat = view_.persmat();
    /* Vertical projection scale from the pure projection matrix (not corrupted
     * by object rotation). For orthographic views winmat[1][1] == 2/height. */
    const float viewport_proj_scale_y = std::abs(default_view.winmat()[1][1]);
    /* World-space frustum planes for per-instance camera culling, extracted
     * from the pure view-projection (world -> clip) — never from an
     * object-baked matrix. */
    const tessera::TesseraViewInfo tessera_cull_view = tessera::getViewInfo(
        unjittered_persmat, float(resolution.y), viewport_proj_scale_y);

    auto now_frame = std::chrono::steady_clock::now();
    bool view_moved = false;
    if (this->draw_ctx != nullptr && this->draw_ctx->rv3d != nullptr) {
      if (tessera_has_last_view_persmat_ && unjittered_persmat != tessera_last_view_persmat_) {
        view_moved = true;
        tessera_last_view_move_time_ = now_frame;
      }
      tessera_last_view_persmat_ = unjittered_persmat;
      tessera_has_last_view_persmat_ = true;
      if (!view_moved && tessera_last_view_move_time_.time_since_epoch().count() != 0) {
        auto ms_since_move = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now_frame - tessera_last_view_move_time_)
                                 .count();
        if (ms_since_move < 200) {
          view_moved = true;
        }
      }
    }

    Map<std::string, int> drawn_instances_by_emitter;
    auto t_tdraw_start = std::chrono::high_resolution_clock::now();

    /* Pass 1 records all compute traversals; pass 2 issues all indirect draws. Interleaving
     * compute and graphics per object forced Vulkan to end/restart the rendering scope and
     * insert compute->graphics barriers once per object. */
    struct TesseraPendingDraw {
      TesseraObjectState *state;
      bool allow_dna_write;
    };
    Vector<TesseraPendingDraw> pending_draws;

    for (auto item : tessera_objects_.items()) {
      TesseraObjectState &state = *item.value;
      if (!state.used || state.data.clusters.empty()) {
        continue;
      }
      /* Pure Geometry-Nodes instancer base: the native pipeline draws nothing
       * for the base object (its evaluated geometry is an instance cloud),
       * so Tessera must not draw it either. */
      if (state.is_instancer_base) {
        continue;
      }
      /* Never dereference the pointer stored at sync time: the depsgraph may
       * have freed the evaluated object in between (procedural instancers and
       * scatter systems recreate objects every frame). Re-resolve by name. */
      Object *ob = (this->draw_ctx != nullptr) ?
                       tessera_object_revalidate(this->draw_ctx, state.ob_name) :
                       nullptr;
      if (ob == nullptr) {
        if (this->draw_ctx != nullptr) {
          /* The object no longer exists; retire its cached render state. */
          state.used = false;
          state.ob = nullptr;
          state.pass.reset();
          state.band_passes.clear();
          state.data = tessera::TesseraData{};
        }
        continue;
      }
      state.ob = ob;
      if (!is_tessera_eligible_mesh(ob)) {
        continue;
      }
      Object *arch_ob = (!state.matched_archetype_name.empty() && this->draw_ctx != nullptr) ?
                            tessera_object_revalidate(this->draw_ctx, state.matched_archetype_name) :
                            nullptr;

      /* Rasterization matrix: always projects to the active viewport framebuffer. */
      const float4x4 object_persmat = viewport_persmat * state.object_mat;
      /* Unjittered viewport matrix for stable LOD & frustum culling across TAA samples. */
      const float4x4 unjittered_object_persmat = unjittered_persmat * state.object_mat;

      const int view_source = tessera_view_source_get(ob, arch_ob);
      state.view_source = view_source;

      float4x4 cam_persmat;
      float cam_proj_scale_y = viewport_proj_scale_y;
      const bool in_user_perspective = (this->draw_ctx != nullptr &&
                                        this->draw_ctx->rv3d != nullptr &&
                                        this->draw_ctx->rv3d->persp != RV3D_CAMOB);
      const bool has_explicit_cull_cam = (tessera_property_find_safe(ob, "tessera_cull_camera") !=
                                          nullptr) ||
                                         (arch_ob != nullptr && tessera_property_find_safe(arch_ob, "tessera_cull_camera") != nullptr);
      const bool is_cam_view = in_user_perspective ?
                                   has_explicit_cull_cam :
                                   ((state.view_source != 0) ||
                                    (this->draw_ctx != nullptr && this->draw_ctx->rv3d != nullptr &&
                                     this->draw_ctx->rv3d->persp == RV3D_CAMOB));
      bool has_cam_lod = false;
      if (is_cam_view) {
        if (tessera_camera_persmat(
                scene_state_.scene, this->draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y, arch_ob)) {
          has_cam_lod = true;
        }
      }
      const float4x4 lod_object_persmat = has_cam_lod ? (cam_persmat * state.object_mat) :
                                                        unjittered_object_persmat;
      const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;

      const float culling_margin = tessera_culling_margin_get(scene_state_.scene, state.ob, arch_ob);
      state.culling_margin = culling_margin;
      const float lod_distance = (state.ob != nullptr) ?
          tessera_lod_distance_get(scene_state_.scene, state.ob) : state.lod_distance;
      state.lod_distance = lod_distance;
      const float lod_intensity = (state.ob != nullptr) ?
          tessera_lod_intensity_get(scene_state_.scene, state.ob) : state.lod_intensity;
      state.lod_intensity = lod_intensity;
      const float pixel_error = (state.ob != nullptr) ?
          tessera_pixel_error_get(state.ob) : state.pixel_error;
      state.pixel_error = pixel_error;
      /* Scattered instance fields use a dedicated, coarser LOD threshold:
       * each dupli is tiny on screen, so a coarser cluster cut per instance
       * keeps the rasterized cluster-instance count sane for 100k+ fields. */
      const float effective_pixel_error = (state.instance_mats.size() > 1) ?
          tessera_scatter_pixel_error_get(scene_state_.scene, state.ob, pixel_error) :
          (state.data.clusters.size() > 5000 ? std::max(pixel_error, 8.0f) : pixel_error);

      tessera::TesseraShadingMode mode = tessera::TESSERA_SHADING_DIFFUSE;
      if (state.visualize_clusters == 1) {
        mode = tessera::TESSERA_SHADING_CLUSTERS;
      }
      else if (state.visualize_clusters == 2) {
        mode = tessera::TESSERA_SHADING_NORMALS;
      }
      else if (state.visualize_clusters == 3) {
        mode = tessera::TESSERA_SHADING_TANGENTS;
      }
      else if (state.visualize_clusters == 4) {
        mode = tessera::TESSERA_SHADING_UVS;
      }

      std::vector<tessera::TesseraMaterialTextures> mats;
      if (mode == tessera::TESSERA_SHADING_DIFFUSE) {
        mats.resize(1);
        if (scene_state_.shading.color_type == V3D_SHADING_SINGLE_COLOR) {
          mats[0].base_color = float4(scene_state_.shading.single_color[0],
                                      scene_state_.shading.single_color[1],
                                      scene_state_.shading.single_color[2],
                                      1.0f);
        }
        else if (scene_state_.shading.color_type == V3D_SHADING_OBJECT_COLOR && state.ob != nullptr) {
          mats[0].base_color = float4(state.ob->color[0], state.ob->color[1], state.ob->color[2], 1.0f);
        }
        else if (state.base_color_valid) {
          /* Material color of the instanced archetype: scattered foliage's
           * emitter proxy carries no materials, the real plant does. */
          mats[0].base_color = state.base_color;
        }
        else {
          mats[0].base_color = float4(0.8f, 0.8f, 0.8f, 1.0f);
        }
        if (scene_state_.shading.light == V3D_LIGHTING_FLAT) {
          mats[0].roughness_value = 1.0f;
          mats[0].specular_value = 0.0f;
        }
        else {
          mats[0].roughness_value = 0.5f;
          mats[0].specular_value = 0.5f;
        }
      }

      /* Common per-pass configuration, applied to the main pass and to every
       * distance-band sub-pass of scattered fields. */
      const auto configure_pass = [&](blender::tessera::TesseraDrawPass &pass, int band_idx = 0) {
        pass.setReverseZ(false);
        pass.setUseElementsIndirect(true);
        pass.setCullingMargin(culling_margin);
        /* Adaptive error scaling per distance band:
         * Bands 1 through 7 scale up pixel error progressively,
         * forcing mid and distant foliage/rocks to clamp to coarse parent clusters. */
        float band_err = effective_pixel_error;
        if (band_idx >= 1) {
          band_err *= (1.0f + float(band_idx) * 1.5f);
        }
        pass.setPixelError(band_err);
        pass.setLodDistance(lod_distance);
        pass.setLodIntensity(lod_intensity);
        pass.setShadingMode(mode);
        if (mode == tessera::TESSERA_SHADING_DIFFUSE) {
          pass.set_material_textures(mats);
        }
      };

      if (state.pass == nullptr) {
        state.pass = std::make_unique<blender::tessera::TesseraDrawPass>(
            state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
        if (!state.pass->init()) {
          state.pass.reset();
          continue;
        }
      }
      configure_pass(*state.pass, 0);

      const bool show_stats = state.show_stats;
      const bool log_console = state.log_console;
      const bool show_fps_panel = state.show_fps_panel;
      const bool show_fps_hud = state.show_fps_hud;

      bool is_navigating = view_moved;
      if (this->draw_ctx != nullptr && this->draw_ctx->rv3d != nullptr) {
        is_navigating = is_navigating || this->draw_ctx->is_navigating() ||
                        this->draw_ctx->is_playback() ||
                        ((this->draw_ctx->rv3d->rflag & RV3D_NAVIGATING) != 0) ||
                        (this->draw_ctx->rv3d->viewlock != 0) ||
                        (this->draw_ctx->rv3d->sms != nullptr) ||
                        (this->draw_ctx->rv3d->smooth_timer != nullptr);
      }
      if (scene_state_.scene != nullptr) {
        if (IDProperty *sgroup = IDP_GetProperties(&scene_state_.scene->id)) {
          if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
            if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
              is_navigating = true;
            }
          }
        }
      }
      bool allow_dna_write = false;
      if (!is_navigating) {
        if (this->draw_ctx == nullptr || this->draw_ctx->rv3d == nullptr) {
          allow_dna_write = true;
        }
        else {
          auto now_dna = std::chrono::steady_clock::now();
          if (state.was_navigating || state.last_dna_write_time.time_since_epoch().count() == 0) {
            allow_dna_write = true;
            state.last_dna_write_time = now_dna;
          }
          else {
            auto elapsed_dna_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      now_dna - state.last_dna_write_time)
                                      .count();
            if (elapsed_dna_ms >= 500) {
              allow_dna_write = true;
              state.last_dna_write_time = now_dna;
            }
          }
        }
      }
      state.was_navigating = is_navigating;
      if (allow_dna_write && state.ob != nullptr) {
        tessera_stat_culling_margin_write(state.ob, culling_margin);
      }

      /* Same LOD threshold while navigating and at rest: scaling it during navigation made the
       * whole cluster cut pop (visible blinking) on every navigation start/stop. */
      const auto set_frame_flags = [&](blender::tessera::TesseraDrawPass &pass) {
        pass.setNavigating(is_navigating);
        /* GPU counter readbacks stall the CPU on the whole GPU queue (Vulkan timeline wait);
         * only request them on still frames. */
        pass.setTelemetryEnabled(!is_navigating && (show_stats || log_console ||
                                                    show_fps_hud || show_fps_panel));
      };
      set_frame_flags(*state.pass);

      /* For scattered instance fields, expand the traversal culling margin so
       * the cluster selection is not gated by the base archetype transform
       * (traversal only sees one instance matrix). The traversal planes are
       * transformed into the baked instance's local space — a space that can
       * be scaled/rotated arbitrarily relative to the world — so the
       * worst-case plane distance is bounded by the *local* camera-to-field
       * distance. The per-instance frustum test below performs the real
       * visibility work for every scattered copy at draw time. */
      if (state.instance_mats.size() > 1) {
        float3 field_bs_center(0.0f);
        float field_bs_radius = 0.0f;
        const bool have_bs = tessera::mesh_bounds_sphere(state.data, field_bs_center,
                                                         field_bs_radius);
        const tessera::TesseraViewInfo instance_cull_view = has_cam_lod ?
            tessera::getViewInfo(cam_persmat, float(resolution.y), cam_proj_scale_y) :
            tessera_cull_view;

        if (have_bs) {
          const float4x4 inv_object_mat = math::invert(state.object_mat);
          const float3 cam_pos_local = (inv_object_mat *
                                        float4(instance_cull_view.camPos, 1.0f)).xyz();
          const float dist_cam_local = math::distance(cam_pos_local, field_bs_center);
          state.pass->setCullingMargin(dist_cam_local + 2.0f * field_bs_radius +
                                       state.culling_margin);
        }
        const float max_draw_dist = tessera_max_draw_distance_get(
            scene_state_.scene, state.ob, state.ob_name, state.matched_archetype_name);

        const float min_screen_radius_px = tessera_min_screen_radius_get(
            scene_state_.scene, state.ob, state.matched_archetype_name);
        const float px_scale = (has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y) * (float(resolution.y) * 0.5f);

        /* Per-instance camera frustum, distance & sub-pixel culling (pass 1):
         * Hierarchically reject outside/sub-pixel instances through the BVH;
         * pass 2 rasterizes all surviving instances via a single unified indirect draw. */
        state.frame_visible_instances.clear();
        if (have_bs && field_bs_radius > 0.0f) {
          if (!state.instance_bvh.is_built() ||
              state.instance_bvh.built_instance_count != state.instance_mats.size())
          {
            state.instance_bvh.build(state.instance_mats, field_bs_center, field_bs_radius);
          }
          state.instance_bvh.cull(instance_cull_view,
                                  state.instance_mats,
                                  field_bs_center,
                                  field_bs_radius,
                                  state.culling_margin,
                                  state.frame_visible_instances,
                                  max_draw_dist,
                                  min_screen_radius_px,
                                  px_scale);
        }
        else {
          /* Fallback linear culling when bounding sphere is missing */
          const float max_d_sq = max_draw_dist > 0.0f ? max_draw_dist * max_draw_dist : 0.0f;
          for (const float4x4 &mat : state.instance_mats) {
            const float3 c = (mat * float4(field_bs_center, 1.0f)).xyz();
            const float d_sq = math::distance_squared(c, instance_cull_view.camPos);
            if (max_d_sq > 0.0f && d_sq > max_d_sq) {
              continue;
            }
            if (min_screen_radius_px > 0.0f && px_scale > 0.0f && field_bs_radius > 0.0f) {
              const float d = std::sqrt(std::max(0.0f, d_sq));
              const float inst_scale = std::max(
                  math::length(mat[0].xyz()),
                  std::max(math::length(mat[1].xyz()), math::length(mat[2].xyz())));
              const float world_radius = field_bs_radius * inst_scale;
              const float screen_radius_px = (world_radius * px_scale) / std::max(d, 0.01f);
              if (screen_radius_px < min_screen_radius_px) {
                continue;
              }
            }
            if (instance_in_frustum(instance_cull_view, mat, field_bs_center, field_bs_radius, state.culling_margin)) {
              state.frame_visible_instances.push_back(mat);
            }
          }
        }

        /* Distance-Band Distribution & Clutter Thinning:
         * Partition visible instances into 3 discrete distance bands:
         * Band 0 (< 6m): high detail for close-up inspection.
         * Band 1 (6-14m): mid-distance simplification.
         * Band 2 (>= 14m): coarse simplification for distant scatter. */
        state.frame_band_count = 0;
        for (int k = 0; k < 8; ++k) {
          state.frame_band_instances[k].clear();
        }
        std::array<float, 8> band_nearest_d;
        std::array<float4x4, 8> band_nearest_mat;
        band_nearest_d.fill(std::numeric_limits<float>::max());

        if (!state.frame_visible_instances.empty()) {
          const float3 band_cam_pos = instance_cull_view.camPos;
          const bool is_scatter = state.instance_mats.size() > 1;

          if (is_scatter) {
            const std::string &chk_name = !state.matched_archetype_name.empty() ?
                                              state.matched_archetype_name : state.ob_name;
            const bool is_tiny_clutter = (min_screen_radius_px >= 4.0f) ||
                (chk_name.find("Rock") != std::string::npos && chk_name.find("Small") != std::string::npos) ||
                (chk_name.find("Violet") != std::string::npos) ||
                (chk_name.find("Debris") != std::string::npos) ||
                (chk_name.find("Stick") != std::string::npos);

            const bool is_dense_foliage = (chk_name.find("Galetta") != std::string::npos) ||
                (chk_name.find("Grass") != std::string::npos) ||
                (chk_name.find("DewBerry") != std::string::npos) ||
                (chk_name.find("StickyWilly") != std::string::npos) ||
                (chk_name.find("StingingNettle") != std::string::npos);

            for (size_t inst_idx = 0; inst_idx < state.frame_visible_instances.size(); ++inst_idx) {
              float4x4 inst_mat = state.frame_visible_instances[inst_idx];
              const float3 c = (inst_mat * float4(field_bs_center, 1.0f)).xyz();
              const float d = math::distance(c, band_cam_pos);

              /* Distance thinning for distant tiny clutter & dense foliage:
               * Thin out dense instances past 6m/12m using a stable hash.
               * Visually indistinguishable at distance with scale compensation. */
              if (is_tiny_clutter && d > 6.0f) {
                const uint32_t hash_val = uint32_t(inst_idx * 2654435761u ^
                                                   (uint32_t(std::abs(c.x * 100.0f)) * 2246822519u));
                if (d > 12.0f) {
                  if ((hash_val & 3u) != 0) {
                    continue; // 75% drop
                  }
                  inst_mat[0] *= 1.35f; inst_mat[1] *= 1.35f; inst_mat[2] *= 1.35f;
                }
                else {
                  if ((hash_val & 1u) != 0) {
                    continue; // 50% drop
                  }
                  inst_mat[0] *= 1.20f; inst_mat[1] *= 1.20f; inst_mat[2] *= 1.20f;
                }
              }
              else if (is_dense_foliage && d > 6.0f) {
                const uint32_t hash_val = uint32_t(inst_idx * 2654435761u ^
                                                   (uint32_t(std::abs(c.x * 100.0f)) * 2246822519u));
                if (d > 12.0f) {
                  if ((hash_val & 3u) != 0) {
                    continue; // 75% drop
                  }
                  inst_mat[0] *= 1.35f; inst_mat[1] *= 1.35f; inst_mat[2] *= 1.35f;
                }
                else {
                  if ((hash_val & 1u) != 0) {
                    continue; // 50% drop
                  }
                  inst_mat[0] *= 1.20f; inst_mat[1] *= 1.20f; inst_mat[2] *= 1.20f;
                }
              }

              const int k = 0;

              state.frame_band_instances[k].push_back(inst_mat);
              if (d < band_nearest_d[k]) {
                band_nearest_d[k] = d;
                band_nearest_mat[k] = inst_mat;
              }
            }

            for (int k = 0; k < 3; ++k) {
              if (!state.frame_band_instances[k].empty()) {
                state.frame_band_nearest_d[k] = band_nearest_d[k];
                state.frame_band_count = std::max(state.frame_band_count, uint32_t(k + 1));
              }
            }

            /* debug stat logging removed */
          }
          else {
            const float3 c = (state.frame_visible_instances[0] * float4(field_bs_center, 1.0f)).xyz();
            const float d = math::distance(c, band_cam_pos);
            band_nearest_d[0] = d;
            band_nearest_mat[0] = state.frame_visible_instances[0];
            state.frame_band_nearest_d[0] = d;
            state.frame_band_instances[0] = std::move(state.frame_visible_instances);
            state.frame_band_count = 1;
          }
        }
        /* One LOD traversal per band (band 0 reuses the state's main pass;
         * bands 1..N get their own sub-passes). Each band's selection is
         * re-traversed only when its reference matrix, LOD view or instance
         * count changes. */
        const float4x4 lod_persmat = has_cam_lod ? cam_persmat : unjittered_persmat;
        for (uint32_t k = 0; k < state.frame_band_count; ++k) {
          std::vector<float4x4> &band = state.frame_band_instances[k];
          if (band.empty()) {
            continue;
          }
          blender::tessera::TesseraDrawPass *band_pass = state.pass.get();
          if (k > 0) {
            if (state.band_passes.size() < k) {
              state.band_passes.resize(k);
            }
            if (state.band_passes[k - 1] == nullptr) {
              state.band_passes[k - 1] =
                  std::make_unique<blender::tessera::TesseraDrawPass>(
                      state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
              if (!state.band_passes[k - 1]->init()) {
                state.band_passes[k - 1].reset();
                continue;
              }
            }
            band_pass = state.band_passes[k - 1].get();
          }
          configure_pass(*band_pass, int(k));
          set_frame_flags(*band_pass);
          if (k > 0) {
            band_pass->setTelemetryEnabled(false);
          }
          /* The band's nearest instance is the reference transform: its LOD
           * demand is the strongest in the band, so evaluating the cut there
           * never under-refines any instance of the band. */
          const float4x4 &band_ref = band_nearest_mat[k];
          if (have_bs) {
            const float4x4 inv_ref = math::invert(band_ref);
            const float3 cam_local = (inv_ref * float4(instance_cull_view.camPos, 1.0f)).xyz();
            const float dist_local = math::distance(cam_local, field_bs_center);
            band_pass->setCullingMargin(dist_local + 2.0f * field_bs_radius +
                                        state.culling_margin);
          }
          const float4x4 band_object_persmat = viewport_persmat * band_ref;
          const float4x4 band_lod_object_persmat = lod_persmat * band_ref;
          /* Errors are stored in the source mesh's units; the reference
           * instance's scale converts them to world space so the projected
           * screen error (and thus the LOD cut) is correct. */
          const float band_error_scale = std::max(
              math::length(band_ref[0].xyz()),
              std::max(math::length(band_ref[1].xyz()), math::length(band_ref[2].xyz())));
          band_pass->update(state.data,
                            band_object_persmat,
                            viewport_proj_scale_y,
                            float(resolution.y),
                            &band_lod_object_persmat,
                            &lod_proj_scale_y,
                            uint32_t(band.size()),
                            band_error_scale);
        }
        if (state.frame_band_count == 0) {
          /* All instances of this archetype were culled by frustum/distance culling.
           * Skip GPU compute traversal completely and do not queue for rasterization. */
          if (allow_dna_write && state.ob != nullptr) {
            drawn_instances_by_emitter.lookup_or_add(state.ob_name, 0) += 0;
            tessera_stat_culling_margin_write(state.ob, state.culling_margin);
            if (arch_ob != nullptr && arch_ob != state.ob) {
              tessera_stat_culling_margin_write(arch_ob, state.culling_margin);
              tessera_stat_int_write(arch_ob, "tessera_stat_drawn_instances", 0);
              tessera_stat_int_write(arch_ob, "tessera_stat_clusters", 0);
              tessera_stat_int_write(arch_ob, "tessera_stat_triangles", 0);
            }
          }
          continue;
        }
      }
      else {
        state.frame_visible_instances.clear();
        const float obj_error_scale = std::max(
            math::length(state.object_mat[0].xyz()),
            std::max(math::length(state.object_mat[1].xyz()),
                     math::length(state.object_mat[2].xyz())));
        state.pass->update(state.data,
                           object_persmat,
                           viewport_proj_scale_y,
                           float(resolution.y),
                           &lod_object_persmat,
                           &lod_proj_scale_y,
                           1,
                           obj_error_scale);
      }

      if (show_fps_panel || show_fps_hud) {
        auto now = std::chrono::steady_clock::now();
        if (state.last_frame_time.time_since_epoch().count() == 0) {
          state.last_frame_time = now;
        }
        else {
          float dt = std::chrono::duration<float>(now - state.last_frame_time).count();
          state.last_frame_time = now;
          if (dt > 0.0f && dt < 1.0f) {
            state.smoothed_dt = (state.smoothed_dt <= 0.0f) ? dt : (0.9f * state.smoothed_dt + 0.1f * dt);
          }
        }
        float fps = (state.smoothed_dt > 1e-5f) ? (1.0f / state.smoothed_dt) : 0.0f;
        float frame_time_ms = state.smoothed_dt * 1000.0f;
        if (allow_dna_write && state.ob != nullptr) {
          tessera_write_fps_to_object(state.ob, fps, frame_time_ms);
        }
      }
      else {
        state.last_frame_time = {};
        state.smoothed_dt = 0.0f;
      }

      /* A whole-mesh frustum-culled object has exact (zero) stats without any GPU readback, so
       * publish them even during navigation. */
      const bool tessera_culled_out = (state.pass->drawCount() == 0);
      if ((!is_navigating || tessera_culled_out) && (show_stats || log_console || show_fps_hud)) {
        if (allow_dna_write || log_console || tessera_culled_out) {
          const float3 cam_pos = default_view.location();
          const float3 bounds_min(state.data.header.boundsMin[0],
                                  state.data.header.boundsMin[1],
                                  state.data.header.boundsMin[2]);
          const float3 bounds_max(state.data.header.boundsMax[0],
                                  state.data.header.boundsMax[1],
                                  state.data.header.boundsMax[2]);
          const float3 local_center = 0.5f * (bounds_min + bounds_max);
          const float3 world_center = (state.object_mat * float4(local_center, 1.0f)).xyz();
          const float cam_distance = math::distance(cam_pos, world_center);

          tessera::TesseraTelemetry telem = state.pass->getTelemetry();
          telem.cam_distance = cam_distance;
          telem.pixel_error = state.pass->getPixelError();
          telem.culling_margin = state.culling_margin;

          tessera::TesseraMemoryBudget budget = state.pass ? state.pass->getMemoryBudget() :
                                                             tessera::TesseraMemoryBudget{};
          int vram_budget_mb = 1024;
          if (scene_state_.scene != nullptr) {
            if (IDProperty *group = IDP_GetProperties(&scene_state_.scene->id)) {
              if (IDProperty *prop = IDP_GetPropertyFromGroup(group, "tessera_vram_budget_mb")) {
                if (prop->type == IDP_INT) {
                  vram_budget_mb = IDP_int_get(prop);
                }
              }
            }
          }
          if (vram_budget_mb < 128) {
            vram_budget_mb = 128;
          }
          if (vram_budget_mb > 16384) {
            vram_budget_mb = 16384;
          }
          if (state.pass) {
            state.pass->setBudget(size_t(vram_budget_mb) * 1024ULL * 1024ULL);
          }
          uint32_t total_instances = uint32_t(tessera_objects_.size());
          uint32_t active_instances = 0;
          for (const auto it : tessera_objects_.items()) {
            if (it.value->used && !it.value->data.clusters.empty()) {
              active_instances++;
            }
          }
          if (state.pass) {
            state.pass->setInstanceCounts(total_instances, active_instances);
          }
          budget = state.pass ? state.pass->getMemoryBudget() : budget;

          if ((allow_dna_write || tessera_culled_out) && (show_stats || show_fps_hud) &&
              state.ob != nullptr)
          {
            tessera_write_telemetry_to_object(state.ob, telem, &budget, total_instances, active_instances);
            if (arch_ob != nullptr && arch_ob != state.ob) {
              tessera_write_telemetry_to_object(arch_ob, telem, &budget, total_instances, active_instances);
            }
            if (scene_state_.scene != nullptr) {
              if (IDProperty *sgroup = IDP_EnsureProperties(&scene_state_.scene->id)) {
                idp_float_replace(sgroup, "tessera_stat_vram_resident_mb", float(budget.current_resident_bytes) / (1024.0f * 1024.0f));
                idp_float_replace(sgroup, "tessera_stat_vram_budget_mb", float(budget.max_vram_bytes) / (1024.0f * 1024.0f));
                idp_float_replace(sgroup, "tessera_stat_disk_read_mbs", budget.disk_read_mbs);
                idp_float_replace(sgroup, "tessera_stat_cache_hit_ratio", budget.cache_hit_ratio);
                idp_int_replace(sgroup, "tessera_stat_total_instances", int(total_instances));
                idp_int_replace(sgroup, "tessera_stat_active_instances", int(active_instances));
              }
            }
          }

          if (log_console) {
            printf("[Tessera Stats] %s | Dist: %.2fm | LOD: %u-%u | Clusters: %u/%u | Tris: %u/%u | Error: %.1fpx\n",
                   state.ob ? (state.ob->id.name + 2) : "Object",
                   telem.cam_distance,
                   telem.active_lod_min,
                   telem.active_lod_max,
                   telem.active_clusters,
                   telem.total_clusters,
                   telem.active_triangles,
                   telem.total_triangles,
                   telem.pixel_error);
            fflush(stdout);
          }
        }
      }
      if (state.pass->drawCount() != 0) {
        pending_draws.append({&state, allow_dna_write});
      }
    }
    auto t_pass1_end = std::chrono::high_resolution_clock::now();

    /* Terrain/Solid-first draw ordering: Render non-scattered / single-instance
     * meshes (such as ground and terrain) first to seed the depth buffer. This
     * maximizes GPU Early-Z hardware occlusion rejection for dense scattered
     * foliage cards and rocks. */
    std::stable_sort(pending_draws.begin(), pending_draws.end(),
                     [](const TesseraPendingDraw &a, const TesseraPendingDraw &b) {
                       const bool a_single = (a.state->instance_mats.size() <= 1);
                       const bool b_single = (b.state->instance_mats.size() <= 1);
                       if (a_single != b_single) {
                         return a_single > b_single;
                       }
                       return a.state->instance_mats.size() < b.state->instance_mats.size();
                     });

    for (const TesseraPendingDraw &pending : pending_draws) {
      TesseraObjectState &state = *pending.state;
      const bool allow_dna_write = pending.allow_dna_write;
      Object *arch_ob = (!state.matched_archetype_name.empty() && this->draw_ctx != nullptr) ?
                            tessera_object_revalidate(this->draw_ctx, state.matched_archetype_name) :
                            nullptr;
      {
        int lighting_type = (scene_state_.shading.light == V3D_LIGHTING_FLAT) ? 0 : 1;
        float4 base_col(0.8f, 0.8f, 0.8f, 1.0f);
        if (scene_state_.shading.color_type == V3D_SHADING_SINGLE_COLOR) {
          base_col = float4(scene_state_.shading.single_color[0],
                            scene_state_.shading.single_color[1],
                            scene_state_.shading.single_color[2],
                            1.0f);
        }
        else if (scene_state_.shading.color_type == V3D_SHADING_OBJECT_COLOR && state.ob != nullptr) {
          base_col = float4(state.ob->color[0], state.ob->color[1], state.ob->color[2], 1.0f);
        }
        else if (state.base_color_valid) {
          base_col = state.base_color;
        }
        state.pass->set_two_sided(state.two_sided);
        int drawn_instances = 0;
        if (state.instance_mats.size() > 1) {
          /* Scattered dupli / Geometry-Nodes field: the survivors were
           * frustum-culled and split into distance bands in pass 1 (each band
           * has its own LOD selection); every band is rasterized with a
           * single batched indirect instanced draw (per-instance world
           * matrices from a storage buffer) instead of one draw call per
           * scattered copy. */
          for (uint32_t k = 0; k < state.frame_band_count; ++k) {
            const std::vector<float4x4> &band = state.frame_band_instances[k];
            if (band.empty()) {
              continue;
            }
            blender::tessera::TesseraDrawPass *band_pass =
                (k == 0) ? state.pass.get() :
                           ((k - 1 < state.band_passes.size() &&
                             state.band_passes[k - 1] != nullptr) ?
                                state.band_passes[k - 1].get() :
                                nullptr);
            if (band_pass == nullptr) {
              continue;
            }
            /* Opaque material override for distant foliage (>= 20m):
             * Disabling alpha cutout re-enables full-rate hardware Early-Z depth rejection
             * across thousands of distant leaf cards, eliminating quad-overdraw. */
            const bool allow_alpha = (state.frame_band_nearest_d[k] < 20.0f);
            band_pass->set_alpha_mask(allow_alpha);
            if (band.size() == 1) {
              /* Degenerate single survivor: keep the classic single-matrix
               * path (the GPU args were not instance-multiplied). */
              const float4x4 inst_view_proj = viewport_persmat * band[0];
              band_pass->draw_forward(band[0],
                                      resolution.x,
                                      resolution.y,
                                      lighting_type,
                                      base_col,
                                      &inst_view_proj);
            }
            else {
              band_pass->draw_forward_instances(band.data(),
                                                band.size(),
                                                resolution.x,
                                                resolution.y,
                                                lighting_type,
                                                base_col,
                                                viewport_persmat);
            }
            drawn_instances += int(band.size());
          }
        }
        else {
          state.pass->draw_forward(state.object_mat, resolution.x, resolution.y, lighting_type, base_col);
          drawn_instances = 1;
        }
        /* Expose the number of instances that survived culling for tests and
         * the debug UI (drives the camera-margin verification). Scatter
         * fields split into per-archetype range states: accumulate per
         * emitter and write the sum once below. */
        if (allow_dna_write) {
          if (state.instance_mats.size() > 1) {
            drawn_instances_by_emitter.lookup_or_add(state.ob_name, 0) += drawn_instances;
            if (arch_ob != nullptr && arch_ob != state.ob) {
              tessera_stat_int_write(arch_ob, "tessera_stat_drawn_instances", drawn_instances);
              tessera_stat_culling_margin_write(arch_ob, state.culling_margin);
            }
          }
          else {
            tessera_stat_int_write(state.ob, "tessera_stat_drawn_instances", drawn_instances);
          }
        }
      }
    }



    /* Aggregated drawn-instance counts: one write per emitter object so the
     * stat always reflects the whole scatter field across all archetypes. */
    if (!drawn_instances_by_emitter.is_empty()) {
      for (const auto emitter_item : drawn_instances_by_emitter.items()) {
        Object *stat_ob = (this->draw_ctx != nullptr) ?
                              tessera_object_revalidate(this->draw_ctx, emitter_item.key) :
                              nullptr;
        if (stat_ob != nullptr) {
          tessera_stat_int_write(stat_ob, "tessera_stat_drawn_instances", emitter_item.value);
          if (emitter_item.value == 0) {
            /* Every scattered instance was frustum-culled: nothing of this
             * emitter reaches the rasterizer, so report zero active
             * clusters/triangles as well (keeps camera-culling verification
             * meaningful — the selection stat alone would still show the
             * camera-LOD cut). */
            tessera_stat_int_write(stat_ob, "tessera_stat_clusters", 0);
            tessera_stat_int_write(stat_ob, "tessera_stat_triangles", 0);
          }
        }
      }
    }

    size_t tot_cached_instances = 0;
    for (const auto it : tessera_objects_.items()) {
      tot_cached_instances += it.value->instance_mats.size();
    }
    if (tot_cached_instances > 0) {
      tessera_instances_cached_ = true;
      g_workbench_tessera_instances_cached = true;
    }


  }

  /** \} */

  View view_ = {"DefaultView"};

  SceneState scene_state_;

  SceneResources resources_;

  OpaquePass opaque_ps_;
  TransparentPass transparent_ps_;
  TransparentDepthPass transparent_depth_ps_;

  ShadowPass shadow_ps_;
  VolumePass volume_ps_;
  OutlinePass outline_ps_;
  DofPass dof_ps_;
  AntiAliasingPass anti_aliasing_ps_;

  /* An array of nullptr GPUMaterial pointers so we can call DRW_cache_object_surface_material_get.
   * They never get actually used. */
  Vector<GPUMaterial *> dummy_gpu_materials_ = {1, nullptr, {}};

  /* Used to detect any scene data update. */
  uint64_t depsgraph_last_update_ = 0;
  bool scene_updated_ = true;
  bool tessera_instances_cached_ = false;

  const char *hair_buffer_overflow_error_ = nullptr;

 public:
  const DRWContext *draw_ctx = nullptr;

  StringRefNull name_get() final
  {
    return "Workbench";
  }

  Span<const GPUMaterial *> get_dummy_gpu_materials(int material_count)
  {
    if (material_count > dummy_gpu_materials_.size()) {
      dummy_gpu_materials_.resize(material_count, nullptr);
    }
    return dummy_gpu_materials_.as_span().slice(IndexRange(material_count));
  };

  void init() final
  {
    this->draw_ctx = DRW_context_get();
    init(draw_ctx->depsgraph);
  }

  void init(Depsgraph *depsgraph, Object *camera_ob = nullptr)
  {
    this->draw_ctx = DRW_context_get();
    bool scene_updated = assign_if_different(depsgraph_last_update_,
                                             DEG_get_update_count(depsgraph));
    scene_updated_ = scene_updated;
    if (scene_updated) {
      tessera_instances_cached_ = false;
      g_workbench_tessera_instances_cached = false;
    }

    scene_state_.init(this->draw_ctx, scene_updated, camera_ob);
    shadow_ps_.init(scene_state_, resources_);
    resources_.init(scene_state_, this->draw_ctx);

    outline_ps_.init(scene_state_);
    dof_ps_.init(scene_state_, this->draw_ctx);
    anti_aliasing_ps_.init(scene_state_);
  }

  void begin_sync() final
  {
    /* When TAA accumulation is finished, object_sync() returns early and never re-marks Tessera
     * states as used. Sweeping here would then evict every resident Tessera mesh (forcing a full
     * .nc2 reload + GPU re-upload and a visible blink on the next redraw), so skip the sweep. */
    if (!scene_state_.render_finished) {
      std::lock_guard lock(this->tessera_mutex_);
      tessera_last_state_ = nullptr;
      tessera_last_ob_ = nullptr;
      tessera_last_data_ = nullptr;
      if (!tessera_instances_cached_) {
        this->tessera_sync_round_ += 1;
        for (auto item : tessera_objects_.items()) {
          item.value->used = false;
          item.value->last_instance_count = std::max(item.value->last_instance_count,
                                                     item.value->instance_mats.size());
          item.value->instance_mats.clear();
          if (item.value->last_instance_count > 0) {
            item.value->instance_mats.reserve(item.value->last_instance_count);
          }
        }
      }
      else {
        /* Preserve cached instance transforms for stationary / orbiting camera redraws:
         * avoid re-allocating and re-evaluating 120k+ scatter instances every frame. */
        for (auto item : tessera_objects_.items()) {
          if (!item.value->instance_mats.empty()) {
            item.value->used = true;
          }
        }
      }
    }

    resources_.material_buf.clear_and_trim();

    opaque_ps_.sync(scene_state_, resources_);
    transparent_ps_.sync(scene_state_, resources_);
    transparent_depth_ps_.sync(scene_state_, resources_);

    shadow_ps_.sync();
    volume_ps_.sync(resources_);
    outline_ps_.sync(resources_);
    dof_ps_.sync(resources_, this->draw_ctx);
    anti_aliasing_ps_.sync(scene_state_, resources_);

    hair_buffer_overflow_error_ = nullptr;
  }

  void end_sync() final
  {
    resources_.material_buf.push_update();
  }

  Material get_material(ObjectRef ob_ref, eV3DShadingColorType color_type, int slot = 0)
  {
    switch (color_type) {
      case V3D_SHADING_OBJECT_COLOR:
        return Material(*ob_ref.object);
      case V3D_SHADING_RANDOM_COLOR:
        return Material(*ob_ref.object, true);
      case V3D_SHADING_SINGLE_COLOR:
        return scene_state_.material_override;
      case V3D_SHADING_VERTEX_COLOR:
        return scene_state_.material_attribute_color;
      case V3D_SHADING_TEXTURE_COLOR:
        ATTR_FALLTHROUGH;
      case V3D_SHADING_MATERIAL_COLOR:
        if (blender::Material *_mat = BKE_object_material_get_eval(ob_ref.object, slot + 1)) {
          return Material(*_mat);
        }
        ATTR_FALLTHROUGH;
      default:
        return Material(*BKE_material_default_empty());
    }
  }

  void object_sync(ObjectRef &ob_ref, Manager &manager) final
  {
    if (scene_state_.render_finished) {
      return;
    }

    Object *ob = ob_ref.object;
    if (!DRW_object_is_renderable(ob)) {
      return;
    }

    if (this->tessera_is_enabled(scene_state_.scene, ob)) {
      /* 1. Record object into Tessera's instance pipeline only when cache is invalidated */
      if (!tessera_instances_cached_) {
        this->tessera_record(ob_ref);
      }

      /* 2. CRITICAL: Completely bypass standard Workbench mesh submission!
       * DO NOT call:
       *   - this->mesh_sync(...)
       *   - this->geometry_sync(...)
       *   - DRW_shgroup_call_add(...)
       *   - workbench_material_cache_get(...)
       * Early-return immediately so ZERO standard geometry batches are queued. */
      return;
    }

    const ObjectState object_state = ObjectState(this->draw_ctx, scene_state_, resources_, ob);

    bool is_object_data_visible = (DRW_object_visibility_in_active_context(ob) &
                                   OB_VISIBLE_SELF) &&
                                  (ob->dt >= OB_SOLID || draw_ctx->is_scene_render());

    if (!(ob->base_flag & BASE_FROM_DUPLI)) {
      ModifierData *md = BKE_modifiers_findby_type(ob, eModifierType_Fluid);
      if (md && BKE_modifier_is_enabled(scene_state_.scene, md, eModifierMode_Realtime)) {
        FluidModifierData *fmd = reinterpret_cast<FluidModifierData *>(md);
        if (fmd->domain) {
          volume_ps_.object_sync_modifier(manager, resources_, scene_state_, ob_ref, md);

          if (fmd->domain->type == FLUID_DOMAIN_TYPE_GAS) {
            /* Do not draw solid in this case. */
            is_object_data_visible = false;
          }
        }
      }
    }

    ResourceHandleRange emitter_handle = {};

    if (is_object_data_visible) {
      if (object_state.sculpt_pbvh) {
        ResourceHandleRange handle = manager.unique_handle_for_sculpt(ob_ref);
        this->sculpt_sync(ob_ref, handle, object_state);
        emitter_handle = handle;
      }
      else if (ob->type == OB_MESH) {
        ResourceHandleRange handle = manager.unique_handle(ob_ref);
        this->mesh_sync(ob_ref, handle, object_state);
        emitter_handle = handle;
      }
      else if (ob->type == OB_POINTCLOUD) {
        this->pointcloud_sync(manager, ob_ref, object_state);
      }
      else if (ob->type == OB_CURVES) {
        this->curves_sync(manager, ob_ref, object_state);
      }
      else if (ob->type == OB_VOLUME) {
        if (scene_state_.shading.type != OB_WIRE) {
          volume_ps_.object_sync_volume(manager,
                                        resources_,
                                        scene_state_,
                                        ob_ref,
                                        get_material(ob_ref, object_state.color_type).base_color);
        }
      }
    }

    if (ob->type == OB_MESH && ob->modifiers.first != nullptr) {
      for (ModifierData &md : ob->modifiers) {
        if (md.type != eModifierType_ParticleSystem) {
          continue;
        }
        ParticleSystem *psys = (reinterpret_cast<ParticleSystemModifierData *>(&md))->psys;
        if (!DRW_object_is_visible_psys_in_active_context(ob, psys)) {
          continue;
        }
        ParticleSettings *part = psys->part;
        const int draw_as = (part->draw_as == PART_DRAW_REND) ? part->ren_as : part->draw_as;

        if (draw_as == PART_DRAW_PATH) {
          this->hair_sync(manager, ob_ref, emitter_handle, object_state, psys, &md);
        }
      }
    }
  }

  template<typename F>
  void draw_to_mesh_pass(ObjectRef &ob_ref, bool is_transparent, F draw_callback)
  {
    const bool in_front = (ob_ref.object->dtx & OB_DRAW_IN_FRONT) != 0;

    if (scene_state_.xray_mode || is_transparent) {
      if (in_front) {
        draw_callback(transparent_ps_.accumulation_in_front_ps_);
        draw_callback(transparent_depth_ps_.in_front_ps_);
      }
      else {
        draw_callback(transparent_ps_.accumulation_ps_);
        draw_callback(transparent_depth_ps_.main_ps_);
      }
    }
    else {
      if (in_front) {
        draw_callback(opaque_ps_.gbuffer_in_front_ps_);
      }
      else {
        draw_callback(opaque_ps_.gbuffer_ps_);
      }
    }
  }

  void draw_mesh(ObjectRef &ob_ref,
                 Material &material,
                 gpu::Batch *batch,
                 ResourceHandleRange handle,
                 const MaterialTexture *texture = nullptr,
                 bool show_missing_texture = false)
  {
    resources_.material_buf.append(material);
    int material_index = resources_.material_buf.size() - 1;

    if (show_missing_texture && (!texture || !texture->gpu.texture)) {
      texture = &resources_.missing_texture;
    }

    this->draw_to_mesh_pass(ob_ref, material.is_transparent(), [&](MeshPass &mesh_pass) {
      mesh_pass.get_subpass(eGeometryType::MESH, texture).draw(batch, handle, material_index);
    });
  }

  void mesh_sync(ObjectRef &ob_ref, ResourceHandleRange handle, const ObjectState &object_state)
  {
    if (this->tessera_is_enabled(scene_state_.scene, ob_ref.object)) {
      return;
    }

    bool has_transparent_material = false;

    if (object_state.use_per_material_batches) {
      const int material_count = BKE_object_material_used_with_fallback_eval(*ob_ref.object);

      Span<gpu::Batch *> batches;
      if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
        batches = DRW_cache_mesh_surface_texpaint_get(ob_ref.object);
      }
      else {
        batches = DRW_cache_object_surface_material_get(
            ob_ref.object, this->get_dummy_gpu_materials(material_count));
      }

      if (!batches.is_empty()) {
        for (auto i : IndexRange(material_count)) {
          if (batches[i] == nullptr) {
            continue;
          }

          int material_slot = i;
          Material mat = this->get_material(ob_ref, object_state.color_type, material_slot);
          has_transparent_material = has_transparent_material || mat.is_transparent();

          MaterialTexture texture;
          if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
            texture = MaterialTexture(ob_ref.object, material_slot);
          }

          this->draw_mesh(
              ob_ref, mat, batches[i], handle, &texture, object_state.show_missing_texture);
        }
      }
    }
    else {
      gpu::Batch *batch;
      if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
        batch = DRW_cache_mesh_surface_texpaint_single_get(ob_ref.object);
      }
      else if (object_state.color_type == V3D_SHADING_VERTEX_COLOR) {
        if (ob_ref.object->mode & OB_MODE_VERTEX_PAINT) {
          batch = DRW_cache_mesh_surface_vertpaint_get(ob_ref.object);
        }
        else {
          batch = DRW_cache_mesh_surface_sculptcolors_get(ob_ref.object);
        }
      }
      else {
        batch = DRW_cache_object_surface_get(ob_ref.object);
      }

      if (batch) {
        Material mat = this->get_material(ob_ref, object_state.color_type);
        has_transparent_material = has_transparent_material || mat.is_transparent();

        this->draw_mesh(ob_ref, mat, batch, handle, &object_state.image_paint_override);
      }
    }

    if (object_state.draw_shadow) {
      shadow_ps_.object_sync(scene_state_, ob_ref, handle, has_transparent_material);
    }
  }

  void sculpt_sync(ObjectRef &ob_ref, ResourceHandleRange handle, const ObjectState &object_state)
  {
    if (this->tessera_is_enabled(scene_state_.scene, ob_ref.object)) {
      return;
    }

    SculptBatchFeature features = SCULPT_BATCH_DEFAULT;
    if (object_state.color_type == V3D_SHADING_VERTEX_COLOR) {
      features = SCULPT_BATCH_VERTEX_COLOR;
    }
    else if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
      features = SCULPT_BATCH_UV;
    }

    if (object_state.use_per_material_batches) {
      for (SculptBatch &batch : sculpt_batches_get(ob_ref.object, features)) {
        Material mat = this->get_material(ob_ref, object_state.color_type, batch.material_slot);
        if (scene_state_.show_paint_bvh_debug) {
          mat.base_color = batch.debug_color();
        }

        MaterialTexture texture;
        if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
          texture = MaterialTexture(ob_ref.object, batch.material_slot);
        }

        this->draw_mesh(
            ob_ref, mat, batch.batch, handle, &texture, object_state.show_missing_texture);
      }
    }
    else {
      Material mat = this->get_material(ob_ref, object_state.color_type);
      for (SculptBatch &batch : sculpt_batches_get(ob_ref.object, features)) {
        if (scene_state_.show_paint_bvh_debug) {
          mat.base_color = batch.debug_color();
        }

        this->draw_mesh(ob_ref, mat, batch.batch, handle, &object_state.image_paint_override);
      }
    }
  }

  void pointcloud_sync(Manager &manager, ObjectRef &ob_ref, const ObjectState &object_state)
  {
    ResourceHandleRange handle = manager.unique_handle(ob_ref);

    Material mat = this->get_material(ob_ref, object_state.color_type);
    resources_.material_buf.append(mat);
    int material_index = resources_.material_buf.size() - 1;

    this->draw_to_mesh_pass(ob_ref, mat.is_transparent(), [&](MeshPass &mesh_pass) {
      PassMain::Sub &pass =
          mesh_pass.get_subpass(eGeometryType::POINTCLOUD).sub("Point Cloud SubPass");
      gpu::Batch *batch = pointcloud_sub_pass_setup(pass, ob_ref.object);
      pass.draw(batch, handle, material_index);
    });
  }

  void hair_sync(Manager &manager,
                 ObjectRef &ob_ref,
                 ResourceHandle emitter_handle,
                 const ObjectState &object_state,
                 ParticleSystem *psys,
                 ModifierData *md)
  {
    ResourceHandle handle = manager.resource_handle_for_psys(ob_ref,
                                                             ob_ref.object->object_to_world());

    Material mat = this->get_material(ob_ref, object_state.color_type, psys->part->omat - 1);
    MaterialTexture texture;
    if (object_state.color_type == V3D_SHADING_TEXTURE_COLOR) {
      texture = MaterialTexture(ob_ref.object, psys->part->omat - 1);
    }
    resources_.material_buf.append(mat);
    int material_index = resources_.material_buf.size() - 1;

    this->draw_to_mesh_pass(ob_ref, mat.is_transparent(), [&](MeshPass &mesh_pass) {
      PassMain::Sub &pass =
          mesh_pass.get_subpass(eGeometryType::CURVES, &texture).sub("Hair SubPass");
      pass.push_constant("emitter_object_id", int(emitter_handle.raw()));
      gpu::Batch *batch = hair_sub_pass_setup(pass, scene_state_.scene, ob_ref, psys, md);
      pass.draw(batch, {ResourceID(handle)}, material_index);
    });
  }

  void curves_sync(Manager &manager, ObjectRef &ob_ref, const ObjectState &object_state)
  {
    ResourceHandleRange handle = manager.unique_handle(ob_ref);

    Material mat = this->get_material(ob_ref, object_state.color_type);
    resources_.material_buf.append(mat);
    int material_index = resources_.material_buf.size() - 1;

    this->draw_to_mesh_pass(ob_ref, mat.is_transparent(), [&](MeshPass &mesh_pass) {
      PassMain::Sub &pass = mesh_pass.get_subpass(eGeometryType::CURVES).sub("Curves SubPass");

      const char *error = nullptr;
      gpu::Batch *batch = curves_sub_pass_setup(pass, scene_state_.scene, ob_ref.object, error);
      if (error) {
        hair_buffer_overflow_error_ = error;
      }
      pass.draw(batch, handle, material_index);
    });
  }

  void draw(Manager &manager,
            gpu::Texture *depth_tx,
            gpu::Texture *depth_in_front_tx,
            gpu::Texture *color_tx)
  {
    int2 resolution = scene_state_.resolution;

    /** Always setup in-front depth, since Overlays can be updated without causing a Workbench
     * re-sync (See #113580). */
    bool needs_depth_in_front = !transparent_ps_.accumulation_in_front_ps_.is_empty() ||
                                (!opaque_ps_.gbuffer_in_front_ps_.is_empty() &&
                                 scene_state_.sample == 0);
    resources_.depth_in_front_tx.wrap(needs_depth_in_front ? depth_in_front_tx : nullptr);
    if (!needs_depth_in_front || opaque_ps_.gbuffer_in_front_ps_.is_empty()) {
      resources_.clear_in_front_fb.ensure(GPU_ATTACHMENT_TEXTURE(depth_in_front_tx));
      resources_.clear_in_front_fb.bind();
      GPU_framebuffer_clear_depth_stencil(resources_.clear_in_front_fb, 1.0f, 0x00);
    }

    resources_.depth_tx.wrap(depth_tx);
    resources_.color_tx.wrap(color_tx);

    if (scene_state_.render_finished) {
      /* Just copy back the already rendered result */
      anti_aliasing_ps_.draw(
          draw_ctx, manager, View::default_get(), scene_state_, resources_, depth_in_front_tx);
      return;
    }

    anti_aliasing_ps_.setup_view(view_, scene_state_);

    GPUAttachment id_attachment = GPU_ATTACHMENT_NONE;
    if (scene_state_.draw_object_id) {
      resources_.object_id_tx.acquire_2d(resolution,
                                         gpu::TextureFormat::UINT_16,
                                         GPU_TEXTURE_USAGE_SHADER_READ |
                                             GPU_TEXTURE_USAGE_ATTACHMENT);
      id_attachment = GPU_ATTACHMENT_TEXTURE(resources_.object_id_tx);
    }
    resources_.clear_fb.ensure(GPU_ATTACHMENT_TEXTURE(resources_.depth_tx),
                               GPU_ATTACHMENT_TEXTURE(resources_.color_tx),
                               id_attachment);
    resources_.clear_fb.bind();
    std::array<double4, 2> clear_colors = {double4(scene_state_.background_color), double4(0.0f)};
    GPU_framebuffer_multi_clear(resources_.clear_fb, clear_colors);
    GPU_framebuffer_clear_depth_stencil(resources_.clear_fb, 1.0f, 0x00);

    opaque_ps_.draw(
        manager, view_, resources_, resolution, scene_state_.draw_shadows ? &shadow_ps_ : nullptr);
    transparent_ps_.draw(manager, view_, resources_, resolution);
    transparent_depth_ps_.draw(manager, view_, resources_);

    /* Tessera objects (experimental): render through the visibility-buffer
     * pipeline, compositing over the scene depth. */
    resources_.clear_fb.bind();
    this->tessera_draw(resolution);

    volume_ps_.draw(manager, view_, resources_);
    outline_ps_.draw(manager, resources_);
    dof_ps_.draw(manager, view_, resources_, resolution);
    anti_aliasing_ps_.draw(draw_ctx, manager, view_, scene_state_, resources_, depth_in_front_tx);

    resources_.object_id_tx.release();
  }

  void draw_viewport(Manager &manager,
                     gpu::Texture *depth_tx,
                     gpu::Texture *depth_in_front_tx,
                     gpu::Texture *color_tx)
  {
    this->draw(manager, depth_tx, depth_in_front_tx, color_tx);

    if (scene_state_.sample + 1 < scene_state_.samples_len) {
      DRW_viewport_request_redraw();
    }

    if (hair_buffer_overflow_error_) {
      STRNCPY(info, hair_buffer_overflow_error_);
    }
    else {
      STRNCPY(info, "");
    }
  }

  void draw(Manager &manager) final
  {
    DefaultTextureList *dtxl = draw_ctx->viewport_texture_list_get();

    DRW_submission_start();
    if (draw_ctx->is_viewport_image_render()) {
      draw_image_render(manager, dtxl->depth, dtxl->depth_in_front, dtxl->color);
    }
    else {
      draw_viewport(manager, dtxl->depth, dtxl->depth_in_front, dtxl->color);
    }
    DRW_submission_end();
  }

  void draw_image_render(Manager &manager,
                         gpu::Texture *depth_tx,
                         gpu::Texture *depth_in_front_tx,
                         gpu::Texture *color_tx,
                         RenderEngine *engine = nullptr)
  {
    if (scene_state_.render_finished) {
      /* This can happen in viewport animation renders, if the scene didn't have any updates
       * between frames. */
      this->draw(manager, depth_tx, depth_in_front_tx, color_tx);
      return;
    }

    BLI_assert(scene_state_.sample == 0);
    for (auto i : IndexRange(scene_state_.samples_len)) {
      if (hair_buffer_overflow_error_) {
        RE_engine_set_error_message(engine, hair_buffer_overflow_error_);
      }

      if (engine && RE_engine_test_break(engine)) {
        break;
      }
      if (i != 0) {
        scene_state_.sample = i;
        /* Re-sync anything dependent on scene_state.sample. */
        resources_.init(scene_state_, draw_ctx);
        dof_ps_.init(scene_state_, draw_ctx);
        anti_aliasing_ps_.sync(scene_state_, resources_);
      }
      this->draw(manager, depth_tx, depth_in_front_tx, color_tx);
      /* Metal: Perform render step between samples to allow flushing of freed GPUBackend
       * resources. Vulkan: Perform render step between samples to avoid allocation of a high
       * amount of command buffer memory that can eventually result in out-of-memory errors or a
       * TDR when submitted as one large command buffer. */
      if (ELEM(GPU_backend_get_type(), GPU_BACKEND_METAL, GPU_BACKEND_VULKAN)) {
        GPU_flush();
      }
      GPU_render_step();
    }
  }
};

DrawEngine *Engine::create_instance()
{
  return new Instance();
}

void Engine::free_static()
{
  ShaderCache::release();
}

}  // namespace workbench

/* -------------------------------------------------------------------- */
/** \name Interface with legacy C DRW manager
 * \{ */

/* RENDER */

static bool workbench_render_framebuffers_init(const DRWContext *draw_ctx)
{
  /* For image render, allocate own buffers because we don't have a viewport. */
  const float2 viewport_size = draw_ctx->viewport_size_get();
  const int2 size = {int(viewport_size.x), int(viewport_size.y)};

  DefaultTextureList *dtxl = draw_ctx->viewport_texture_list_get();

  /* When doing a multi view rendering the first view will allocate the buffers
   * the other views will reuse these buffers */
  if (dtxl->color == nullptr) {
    BLI_assert(dtxl->depth == nullptr);
    eGPUTextureUsage usage = GPU_TEXTURE_USAGE_GENERAL;
    dtxl->color = GPU_texture_create_2d(
        "txl.color", size.x, size.y, 1, gpu::TextureFormat::SFLOAT_16_16_16_16, usage, nullptr);
    dtxl->depth = GPU_texture_create_2d("txl.depth",
                                        size.x,
                                        size.y,
                                        1,
                                        gpu::TextureFormat::SFLOAT_32_DEPTH_UINT_8,
                                        usage,
                                        nullptr);
    dtxl->depth_in_front = GPU_texture_create_2d("txl.depth_in_front",
                                                 size.x,
                                                 size.y,
                                                 1,
                                                 gpu::TextureFormat::SFLOAT_32_DEPTH_UINT_8,
                                                 usage,
                                                 nullptr);
  }

  if (!(dtxl->depth && dtxl->color && dtxl->depth_in_front)) {
    return false;
  }

  DefaultFramebufferList *dfbl = draw_ctx->viewport_framebuffer_list_get();

  GPU_framebuffer_ensure_config(
      &dfbl->default_fb,
      {GPU_ATTACHMENT_TEXTURE(dtxl->depth), GPU_ATTACHMENT_TEXTURE(dtxl->color)});

  GPU_framebuffer_ensure_config(&dfbl->depth_only_fb,
                                {GPU_ATTACHMENT_TEXTURE(dtxl->depth), GPU_ATTACHMENT_NONE});

  GPU_framebuffer_ensure_config(&dfbl->color_only_fb,
                                {GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(dtxl->color)});

  return GPU_framebuffer_check_valid(dfbl->default_fb, nullptr) &&
         GPU_framebuffer_check_valid(dfbl->color_only_fb, nullptr) &&
         GPU_framebuffer_check_valid(dfbl->depth_only_fb, nullptr);
}

static void write_render_color_output(RenderLayer *layer,
                                      const char *viewname,
                                      gpu::FrameBuffer *fb,
                                      const rcti *rect)
{
  RenderPass *rp = RE_pass_find_by_name(layer, RE_PASSNAME_COMBINED, viewname);
  if (rp) {
    GPU_framebuffer_bind(fb);
    GPU_framebuffer_read_color(fb,
                               rect->xmin,
                               rect->ymin,
                               BLI_rcti_size_x(rect),
                               BLI_rcti_size_y(rect),
                               4,
                               0,
                               GPU_DATA_FLOAT,
                               rp->ibuf->float_data_for_write());
  }
}

static void write_render_z_output(RenderLayer *layer,
                                  const char *viewname,
                                  gpu::FrameBuffer *fb,
                                  const rcti *rect,
                                  const float4x4 &winmat)
{
  RenderPass *rp = RE_pass_find_by_name(layer, RE_PASSNAME_DEPTH, viewname);
  if (rp) {
    GPU_framebuffer_bind(fb);
    GPU_framebuffer_read_depth(fb,
                               rect->xmin,
                               rect->ymin,
                               BLI_rcti_size_x(rect),
                               BLI_rcti_size_y(rect),
                               GPU_DATA_FLOAT,
                               rp->ibuf->float_data_for_write());

    int pix_num = BLI_rcti_size_x(rect) * BLI_rcti_size_y(rect);

    /* Convert GPU depth [0..1] to view Z [near..far] */
    if (draw::View::default_get().is_persp()) {
      for (float &z : MutableSpan(rp->ibuf->float_data_for_write(), pix_num)) {
        if (z == 1.0f) {
          z = 1e10f; /* Background */
        }
        else {
          z = z * 2.0f - 1.0f;
          z = winmat[3][2] / (z + winmat[2][2]);
        }
      }
    }
    else {
      /* Keep in mind, near and far distance are negatives. */
      float near = draw::View::default_get().near_clip();
      float far = draw::View::default_get().far_clip();
      float range = fabsf(far - near);

      for (float &z : MutableSpan(rp->ibuf->float_data_for_write(), pix_num)) {
        if (z == 1.0f) {
          z = 1e10f; /* Background */
        }
        else {
          z = z * range - near;
        }
      }
    }
  }
}

static void workbench_render_to_image(RenderEngine *engine, RenderLayer *layer, const rcti rect)
{
  using namespace blender::draw;
  const DRWContext *draw_ctx = DRW_context_get();

  if (!workbench_render_framebuffers_init(draw_ctx)) {
    RE_engine_report(engine, RPT_ERROR, "Failed to allocate GPU buffers");
    return;
  }

  /* Setup */
  DefaultFramebufferList *dfbl = draw_ctx->viewport_framebuffer_list_get();
  Depsgraph *depsgraph = draw_ctx->depsgraph;

  workbench::Instance instance;

  /* TODO(sergey): Shall render hold pointer to an evaluated camera instead? */
  Object *camera_ob = DEG_get_evaluated(depsgraph, RE_GetCamera(engine->re));

  /* Set the perspective, view and window matrix. */
  float4x4 winmat, viewmat, viewinv;
  RE_GetCameraWindow(engine->re, camera_ob, winmat.ptr());
  RE_GetCameraModelMatrix(engine->re, camera_ob, viewinv.ptr());
  viewmat = math::invert(viewinv);

  /* Render */
  /* TODO: Remove old draw manager calls. */
  DRW_cache_restart();
  draw::View::default_set(float4x4(viewmat), float4x4(winmat));

  instance.init(depsgraph, camera_ob);

  draw::Manager &manager = *DRW_manager_get();
  manager.begin_sync();

  instance.begin_sync();
  DRW_render_object_iter(
      engine,
      depsgraph,
      [&](draw::ObjectRef &ob_ref, RenderEngine * /*engine*/, Depsgraph * /*depsgraph*/) {
        instance.object_sync(ob_ref, manager);
      });
  instance.end_sync();

  manager.end_sync();

  DRW_submission_start();

  DefaultTextureList &dtxl = *draw_ctx->viewport_texture_list_get();
  instance.draw_image_render(manager, dtxl.depth, dtxl.depth_in_front, dtxl.color, engine);

  DRW_submission_end();

  /* Write image */
  const char *viewname = RE_GetActiveRenderView(engine->re);
  write_render_color_output(layer, viewname, dfbl->default_fb, &rect);
  write_render_z_output(layer, viewname, dfbl->default_fb, &rect, winmat);
}

static void workbench_render_update_passes(RenderEngine *engine,
                                           Scene *scene,
                                           ViewLayer *view_layer)
{
  if (view_layer->passflag & SCE_PASS_COMBINED) {
    RE_engine_register_pass(engine, scene, view_layer, RE_PASSNAME_COMBINED, 4, "RGBA", SOCK_RGBA);
  }
  if (view_layer->passflag & SCE_PASS_DEPTH) {
    RE_engine_register_pass(engine, scene, view_layer, RE_PASSNAME_DEPTH, 1, "Z", SOCK_FLOAT);
  }
}

static void workbench_render(RenderEngine *engine, Depsgraph *depsgraph)
{
  DRW_render_to_image(engine, depsgraph, workbench_render_to_image, [](RenderResult *) {});
}

RenderEngineType DRW_engine_viewport_workbench_type = {
    /*next*/ nullptr,
    /*prev*/ nullptr,
    /*idname*/ "BLENDER_WORKBENCH",
    /*name*/ N_("Workbench"),
    /*flag*/ RE_INTERNAL | RE_USE_STEREO_VIEWPORT | RE_USE_GPU_CONTEXT,
    /*update*/ nullptr,
    /*render*/ &workbench_render,
    /*render_frame_finish*/ nullptr,
    /*draw*/ nullptr,
    /*bake*/ nullptr,
    /*view_update*/ nullptr,
    /*view_draw*/ nullptr,
    /*update_script_node*/ nullptr,
    /*update_render_passes*/ &workbench_render_update_passes,
    /*update_custom_camera*/ nullptr,
    /*draw_engine*/ nullptr,
    /*rna_ext*/
    {
        /*data*/ nullptr,
        /*srna*/ nullptr,
        /*call*/ nullptr,
    },
};

/** \} */

}  // namespace blender
