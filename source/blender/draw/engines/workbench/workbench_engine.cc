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
#include "tessera_draw.hh"

#include "workbench_private.hh"

#include "workbench_engine.h" /* Own include. */

#include "BKE_idprop.hh"
#include "BKE_main.hh"
#include "WM_api.hh"
#include "WM_types.hh"

#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace blender {

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
    /* LOD view source: 0 = viewport view, 1 = the scene's active camera. */
    int view_source = 0;
    bool used = false;
    Object *ob = nullptr;
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

  /** Tessera objects found during sync, keyed by object name. */
  Map<std::string, std::unique_ptr<TesseraObjectState>> tessera_objects_;
  /** Guards #tessera_objects_ (object_sync may run on extraction threads). */
  std::mutex tessera_mutex_;

  /* Tessera data ownership helpers. */

  static IDProperty *tessera_property_find_safe(Object *ob, const char *prop_name)
  {
    if (!ob) {
      return nullptr;
    }
    /* Check evaluated ID first */
    if (ob->id.properties && ob->id.properties->type == IDP_GROUP) {
      IDProperty *prop = IDP_GetPropertyFromGroup(ob->id.properties, prop_name);
      if (prop) {
        return prop;
      }
    }
    /* Fallback to original datablock */
    ID *orig_id = DEG_get_original_id(&ob->id);
    if (orig_id && orig_id->properties && orig_id->properties->type == IDP_GROUP) {
      return IDP_GetPropertyFromGroup(orig_id->properties, prop_name);
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
  static bool tessera_enabled_get(Object *ob)
  {
    if (!ob) {
      return false;
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

  static bool tessera_is_enabled(Object *ob)
  {
    if (!is_tessera_eligible_mesh(ob)) {
      return false;
    }
    if (!tessera_enabled_get(ob)) {
      return false;
    }
    std::string dummy;
    return tessera_path_get(ob, dummy);
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

  static float tessera_culling_margin_get(Object *ob)
  {
    if (!ob) {
      return 0.02f;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_culling_margin");
    if (prop == nullptr) {
      return 0.02f;
    }
    float value = 0.02f;
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

  /** LOD view source: 0 = viewport view, 1 = the scene's active camera. */
  static int tessera_view_source_get(Object *ob)
  {
    if (!ob) {
      return 0;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_view_source");
    if (prop == nullptr || prop->type != IDP_INT) {
      return 0; /* Safe default (Viewport Mode). */
    }
    return IDP_int_get(prop);
  }

  static bool tessera_show_stats_get(Object *ob)
  {
    if (!ob) {
      return true;
    }
    const IDProperty *prop = tessera_property_find_safe(ob, "tessera_show_stats");
    if (prop == nullptr) {
      return true;
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

    write_props(&ob->id);
    const ID *orig_id = DEG_get_original_id(&ob->id);
    if (orig_id != nullptr && orig_id != &ob->id) {
      write_props(const_cast<ID *>(orig_id));
    }
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

    write_props(&ob->id);
    const ID *orig_id = DEG_get_original_id(&ob->id);
    if (orig_id != nullptr && orig_id != &ob->id) {
      write_props(const_cast<ID *>(orig_id));
    }
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

    write_props(&ob->id);
    const ID *orig_id = DEG_get_original_id(&ob->id);
    if (orig_id != nullptr && orig_id != &ob->id) {
      write_props(const_cast<ID *>(orig_id));
    }
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
                                    float &r_proj_scale_y)
  {
    if (scene == nullptr || draw_ctx->depsgraph == nullptr) {
      return false;
    }
    const Object *cam_ob = nullptr;
    if (state_ob != nullptr) {
      IDProperty *cull_cam_prop = tessera_property_find_safe(const_cast<Object *>(state_ob), "tessera_cull_camera");
      if (cull_cam_prop != nullptr) {
        if (cull_cam_prop != nullptr && cull_cam_prop->type == IDP_STRING) {
          const char *cam_name = IDP_string_get(cull_cam_prop);
          if (cam_name != nullptr && cam_name[0] != '\0') {
            Main *bmain = DEG_get_bmain(draw_ctx->depsgraph);
            if (bmain != nullptr) {
              Object *found = static_cast<Object *>(
                  BLI_findstring(&bmain->objects, cam_name, offsetof(ID, name) + 2));
              if (found != nullptr) {
                cam_ob = DEG_get_evaluated(draw_ctx->depsgraph, found);
              }
            }
          }
        }
      }
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
  bool tessera_record(Object *ob)
  {
    if (!is_tessera_eligible_mesh(ob)) {
      return false;
    }
    if (!tessera_enabled_get(ob)) {
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

    BLI_stat_t file_info;
    const bool stat_ok = (BLI_stat(abs_path, &file_info) == 0);
    const int64_t file_mtime = stat_ok ? int64_t(file_info.st_mtime) : int64_t(-1);
    const int64_t file_size = stat_ok ? int64_t(file_info.st_size) : int64_t(-1);

    std::lock_guard lock(this->tessera_mutex_);
    std::unique_ptr<TesseraObjectState> &entry = tessera_objects_.lookup_or_add_default(
        std::string(ob->id.name + 2));
    if (entry == nullptr) {
      entry = std::make_unique<TesseraObjectState>();
    }
    TesseraObjectState &state = *entry;

    state.used = true;
    state.ob = ob;
    state.show_stats = tessera_show_stats_get(ob);
    state.log_console = tessera_log_console_get(ob);
    state.show_fps_panel = tessera_show_fps_panel_get(ob);
    state.show_fps_hud = tessera_show_fps_hud_get(ob);
    state.visualize_clusters = tessera_visualize_get(ob);
    state.pixel_error = tessera_pixel_error_get(ob);
    state.view_source = tessera_view_source_get(ob);
    state.culling_margin = tessera_culling_margin_get(ob);
    state.object_mat = ob->object_to_world();

    /* (Re)load the cluster data when the file path, mtime, or size changed. */
    const bool stale = (state.nc2_path != abs_path) ||
                       (state.file_mtime != file_mtime) ||
                       (state.file_size != file_size);
    if (state.pass == nullptr || stale) {
      if (stale || state.data.clusters.empty()) {
        size_t data_size = 0;
        if (void *file_data = BLI_file_read_binary_as_mem(abs_path, 0, &data_size)) {
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
        state.file_mtime = file_mtime;
        state.file_size = file_size;
        state.nc2_path = abs_path;
        /* Drop the pass so it is rebuilt with the new data on draw. */
        state.pass.reset();
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
    const float4x4 viewport_persmat = view_.persmat();
    /* Vertical projection scale from the pure projection matrix (not corrupted
     * by object rotation). For orthographic views winmat[1][1] == 2/height. */
    const float viewport_proj_scale_y = std::abs(default_view.winmat()[1][1]);

    for (auto item : tessera_objects_.items()) {
      TesseraObjectState &state = *item.value;
      if (!state.used || state.data.clusters.empty()) {
        continue;
      }
      /* Never dereference the pointer stored at sync time: the depsgraph may
       * have freed the evaluated object in between (procedural instancers and
       * scatter systems recreate objects every frame). Re-resolve by name. */
      Object *ob = (this->draw_ctx != nullptr) ?
                       tessera_object_revalidate(this->draw_ctx, item.key) :
                       nullptr;
      if (ob == nullptr) {
        if (this->draw_ctx != nullptr) {
          /* The object no longer exists; retire its cached render state. */
          state.used = false;
          state.ob = nullptr;
          state.pass.reset();
        }
        continue;
      }
      state.ob = ob;
      if (!is_tessera_eligible_mesh(ob)) {
        continue;
      }
      /* Rasterization matrix: always projects to the active viewport framebuffer. */
      const float4x4 object_persmat = viewport_persmat * state.object_mat;

      const int view_source = tessera_view_source_get(ob);
      state.view_source = view_source;

      float4x4 cam_persmat;
      float cam_proj_scale_y = viewport_proj_scale_y;
      const bool is_cam_view = (state.view_source != 0) ||
                               (this->draw_ctx != nullptr && this->draw_ctx->rv3d != nullptr &&
                                this->draw_ctx->rv3d->persp == RV3D_CAMOB);
      bool has_cam_lod = false;
      if (is_cam_view) {
        if (tessera_camera_persmat(
                scene_state_.scene, this->draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y)) {
          has_cam_lod = true;
        }
      }
      const float4x4 lod_object_persmat = has_cam_lod ? (cam_persmat * state.object_mat) :
                                                        object_persmat;
      const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;

      if (state.pass == nullptr) {
        state.pass = std::make_unique<blender::tessera::TesseraDrawPass>(
            state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
        if (!state.pass->init()) {
          state.pass.reset();
          continue;
        }
      }

      state.pass->setReverseZ(false);
      const float culling_margin = (state.ob != nullptr) ?
          tessera_culling_margin_get(state.ob) : state.culling_margin;
      state.culling_margin = culling_margin;
      state.pass->setCullingMargin(culling_margin);
      tessera_stat_culling_margin_write(state.ob, culling_margin);
      const float pixel_error = (state.ob != nullptr) ?
          tessera_pixel_error_get(state.ob) : state.pixel_error;
      state.pixel_error = pixel_error;
      state.pass->setPixelError(pixel_error);

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
      state.pass->setShadingMode(mode);

      if (mode == tessera::TESSERA_SHADING_DIFFUSE) {
        std::vector<tessera::TesseraMaterialTextures> mats(1);
        if (scene_state_.shading.color_type == V3D_SHADING_SINGLE_COLOR) {
          mats[0].base_color = float4(scene_state_.shading.single_color[0],
                                      scene_state_.shading.single_color[1],
                                      scene_state_.shading.single_color[2],
                                      1.0f);
        }
        else if (scene_state_.shading.color_type == V3D_SHADING_OBJECT_COLOR && state.ob != nullptr) {
          mats[0].base_color = float4(state.ob->color[0], state.ob->color[1], state.ob->color[2], 1.0f);
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
        state.pass->set_material_textures(mats);
      }

      const bool show_stats = state.show_stats;
      const bool log_console = state.log_console;
      const bool show_fps_panel = state.show_fps_panel;
      const bool show_fps_hud = state.show_fps_hud;
      state.pass->setTelemetryEnabled(show_stats || log_console || show_fps_hud || show_fps_panel);

      bool is_navigating = false;
      if (this->draw_ctx != nullptr && this->draw_ctx->rv3d != nullptr) {
        is_navigating = ((this->draw_ctx->rv3d->rflag & RV3D_NAVIGATING) != 0) || (this->draw_ctx->rv3d->viewlock != 0);
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
      if (this->draw_ctx == nullptr || this->draw_ctx->rv3d == nullptr) {
        allow_dna_write = true;
      }
      else if (!is_navigating) {
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
      state.was_navigating = is_navigating;

      float effective_pixel_error = pixel_error;
      if (is_navigating) {
        effective_pixel_error *= 2.5f;
      }
      state.pass->setPixelError(effective_pixel_error);
      state.pass->setNavigating(is_navigating);

      state.pass->update(state.data,
                         object_persmat,
                         viewport_proj_scale_y,
                         float(resolution.y),
                         has_cam_lod ? &lod_object_persmat : nullptr,
                         has_cam_lod ? &lod_proj_scale_y : nullptr);

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

      if (show_stats || log_console || show_fps_hud) {
        using namespace std::chrono;
        static auto last_stat_time = steady_clock::now();
        auto now = steady_clock::now();
        auto elapsed_ms = duration_cast<milliseconds>(now - last_stat_time).count();
        if (elapsed_ms >= 60) {
          last_stat_time = now;
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

          if (allow_dna_write && (show_stats || show_fps_hud) && state.ob != nullptr) {
            tessera_write_telemetry_to_object(state.ob, telem, &budget, total_instances, active_instances);
            if (!is_navigating) {
              WM_main_add_notifier(NC_OBJECT | ND_DRAW, state.ob);
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
        state.pass->draw_forward(state.object_mat, resolution.x, resolution.y, lighting_type, base_col);
      }
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

    scene_state_.init(this->draw_ctx, scene_updated, camera_ob);
    shadow_ps_.init(scene_state_, resources_);
    resources_.init(scene_state_, this->draw_ctx);

    outline_ps_.init(scene_state_);
    dof_ps_.init(scene_state_, this->draw_ctx);
    anti_aliasing_ps_.init(scene_state_);
  }

  void begin_sync() final
  {
    {
      std::lock_guard lock(this->tessera_mutex_);
      for (auto item : tessera_objects_.items()) {
        item.value->used = false;
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

    if (this->tessera_is_enabled(ob)) {
      /* 1. Record object into Tessera's instance pipeline */
      this->tessera_record(ob);

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
    if (this->tessera_is_enabled(ob_ref.object)) {
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
    if (this->tessera_is_enabled(ob_ref.object)) {
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
