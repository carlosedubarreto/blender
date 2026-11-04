/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * Implementation of Tessera GPU cluster rendering into EEVEE.
 */

#include "eevee_tessera.hh"
#include "eevee_instance.hh"

#include "draw_handle.hh"
#include "tessera_archetype.hh"
#include "tessera_lod.hh"

#include "DNA_camera_types.h"
#include "DNA_ID.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_view3d_types.h"

#include "BKE_camera.h"
#include "BKE_idprop.hh"
#include "BKE_image.hh"
#include "BKE_main.hh"
#include "BKE_material.hh"
#include "BKE_node.hh"
#include "BKE_node_legacy_types.hh"
#include "BLI_listbase.h"

#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_node_types.h"

#include "BLI_fileops.h"
#include "BLI_math_matrix.hh"
#include "BLI_path_utils.hh"
#include "BLI_string.h"

#include "DEG_depsgraph_query.hh"

#include "GPU_framebuffer.hh"

#include "draw_view.hh"
#include "WM_api.hh"
#include "WM_types.hh"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <cstdarg>

namespace blender::eevee {

static bool g_eevee_tessera_instances_cached = false;

/* TEMP DEBUG (Material Preview scatter investigation): enable with env TESSERA_DEBUG=1 */
bool tess_eevee_dbg_enabled()
{
  static const bool enabled = std::getenv("TESSERA_DEBUG") != nullptr;
  return enabled;
}

void tess_eevee_log(const char *fmt, ...)
{
  char buf[1024];
  std::va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::fputs(buf, stdout);
  std::fflush(stdout);
}

void TesseraModule::tess_eevee_dbg_state(const char *stage, const ObjectState &state) const
{
  if (!tess_eevee_dbg_enabled()) {
    return;
  }
  tess_eevee_log("[tess-eevee] %s: name=%s used=%d clusters=%zu inst_mats=%zu frame_vis=%zu trav=%d arch=%s\n",
              stage,
              state.ob_name.c_str(),
              int(state.used),
              state.data.clusters.size(),
              state.instance_mats.size(),
              state.frame_visible_instances.size(),
              int(state.traversed_this_frame),
              state.matched_archetype_name.c_str());
}

bool DRW_eevee_tessera_instances_cached()
{
  return g_eevee_tessera_instances_cached;
}

void DRW_eevee_tessera_instances_cache_invalidate()
{
  g_eevee_tessera_instances_cached = false;
}

TesseraModule::TesseraModule(Instance &inst) : inst_(inst) {}

TesseraModule::~TesseraModule() = default;

void TesseraModule::begin_sync()
{
  this->tessera_last_state_ = nullptr;
  this->tessera_last_ob_ = nullptr;
  this->tessera_last_data_ = nullptr;
  this->tessera_last_sync_round_ = 0;

  std::lock_guard lock(this->mutex_);
  if (!instances_cached_) {
    this->tessera_sync_round_++;
    for (auto item : objects_.items()) {
      item.value->used = false;
      item.value->traversed_this_frame = false;
      item.value->frame_visible_instances.clear();
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
    for (auto item : objects_.items()) {
      item.value->traversed_this_frame = false;
      item.value->frame_visible_instances.clear();
      if (!item.value->instance_mats.empty()) {
        item.value->used = true;
      }
    }
  }
}

void TesseraModule::end_sync()
{
  std::lock_guard lock(this->mutex_);
  size_t tot_cached_instances = 0;
  for (const auto it : objects_.items()) {
    tot_cached_instances += it.value->instance_mats.size();
  }
  if (tess_eevee_dbg_enabled()) {
    tess_eevee_log("[tess-eevee] end_sync: objs=%zu cached_instances=%zu\n", objects_.size(), tot_cached_instances);
    for (const auto it : objects_.items()) {
      tess_eevee_dbg_state("end_sync", *it.value);
    }
  }
  if (tot_cached_instances > 0) {
    instances_cached_ = true;
    g_eevee_tessera_instances_cached = true;
  }
}

bool TesseraModule::has_valid_cache() const
{
  std::lock_guard lock(const_cast<std::mutex &>(this->mutex_));
  if (!instances_cached_ || objects_.is_empty()) {
    return false;
  }
  for (const auto item : objects_.items()) {
    if (item.value->used && !item.value->data.clusters.empty()) {
      return true;
    }
  }
  return false;
}

static float eevee_max_draw_dist_get(const std::string &ob_name, const std::string &arch_name)
{
  auto contains_ci = [](const std::string &haystack, const char *needle) {
    auto it = std::search(
        haystack.begin(), haystack.end(),
        needle, needle + strlen(needle),
        [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); }
    );
    return it != haystack.end();
  };
  auto contains_any = [&](const char *needle) {
    return contains_ci(ob_name, needle) || (!arch_name.empty() && contains_ci(arch_name, needle));
  };

  /* Allow full view distance for scattered assets matching native Blender placement. */
  (void)contains_any;
  return 0.0f;
}

static IDProperty *tessera_property_find_safe(Object *ob, const char *prop_name)
{
  if (!ob) {
    return nullptr;
  }
  /* Check original datablock first (where user edits / Python properties live).
   * The evaluated object may hold a stale IDProperty snapshot taken when the
   * depsgraph was built, which breaks live viewport property updates. */
  ID *orig_id = DEG_get_original_id(&ob->id);
  if (orig_id && orig_id->properties && orig_id->properties->type == IDP_GROUP) {
    IDProperty *prop = IDP_GetPropertyFromGroup(orig_id->properties, prop_name);
    if (prop) {
      return prop;
    }
  }
  /* Fallback to evaluated ID */
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
 * into or drawn through the Tessera pipeline.
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

/**
 * Detect two-sided / foliage-card geometry: explicit per-object IDProperty
 * override (`tessera_two_sided`) wins. The visibility pipeline renders both
 * faces with flipped backface normals either way; the flag additionally
 * forces opaque solid shading for card-like foliage.
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

/* Expand the traversal culling margin for scattered instance fields so the
 * cluster selection is not gated by the base archetype transform: traversal
 * only sees one instance matrix, and the baked instance can be arbitrarily
 * far outside the frustum (e.g. behind the camera) while other instances are
 * inside. The traversal planes live in the baked instance's local space — a
 * space that can be scaled arbitrarily relative to the world — so the
 * worst-case plane distance is bounded by the *local* camera-to-field
 * distance. The per-instance frustum test performs the real visibility work
 * for every scattered copy at draw time. */
static void tessera_expand_traversal_margin(tessera::TesseraDrawPass &pass,
                                            const tessera::TesseraData &data,
                                            const float4x4 &object_mat,
                                            const std::vector<float4x4> &instance_mats,
                                            const tessera::TesseraViewInfo &cull_view,
                                            float culling_margin)
{
  if (instance_mats.size() <= 1) {
    return;
  }
  float3 field_bs_center(0.0f);
  float field_bs_radius = 0.0f;
  if (!tessera::mesh_bounds_sphere(data, field_bs_center, field_bs_radius)) {
    return;
  }
  /* Camera position in the baked instance's local space. */
  const float4x4 inv_object_mat = math::invert(object_mat);
  const float3 cam_pos_local = (inv_object_mat * float4(cull_view.camPos, 1.0f)).xyz();
  const float dist_cam_local = math::distance(cam_pos_local, field_bs_center);
  pass.setCullingMargin(dist_cam_local + 2.0f * field_bs_radius + culling_margin);
}

static bool tessera_path_get(Object *ob, std::string &r_path)
{
  if (!ob) {
    return false;
  }
  const IDProperty *prop = tessera_property_find_safe(ob, "tessera_nc2");
  if (prop != nullptr && prop->type == IDP_STRING) {
    r_path = IDP_string_get(prop);
    if (!r_path.empty()) {
      return true;
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

static bool tessera_enabled_get(const Scene *scene, Object *ob)
{
  if (!ob) {
    return false;
  }
  if (scene != nullptr) {
    if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "tessera_enabled")) {
      const bool sen = (sprop->type == IDP_INT) ? (IDP_int_get(sprop) != 0) : (sprop->data.val != 0);
      if (!sen) {
        if (tess_eevee_dbg_enabled()) {
          tess_eevee_log("[tess-eevee] en-get FAIL scene-en=0 scene=%s\n", scene->id.name);
        }
        return false;
      }
    }
    if (const IDProperty *sprop = tessera_scene_property_find_safe(scene, "tessera_batch_enabled")) {
      const bool sben = (sprop->type == IDP_INT) ? (IDP_int_get(sprop) != 0) : (sprop->data.val != 0);
      if (!sben) {
        if (tess_eevee_dbg_enabled()) {
          tess_eevee_log("[tess-eevee] en-get FAIL scene-batch=0 scene=%s\n", scene->id.name);
        }
        return false;
      }
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
  const IDProperty *prop = tessera_property_find_safe(ob, "tessera_enabled");
  if (prop != nullptr) {
    const bool val = (prop->type == IDP_INT) ? (IDP_int_get(prop) != 0) : (prop->data.val != 0);
    if (tess_eevee_dbg_enabled() && !val) {
      tess_eevee_log("[tess-eevee] en-get FAIL ob-prop=0 name=%s scene=%s\n", ob->id.name,
                  scene ? scene->id.name : "(null)");
    }
    return val;
  }
  std::string path;
  const bool path_ok = tessera_path_get(ob, path) && !path.empty();
  if (tess_eevee_dbg_enabled() && !path_ok) {
    tess_eevee_log("[tess-eevee] en-get FAIL no-prop-no-path name=%s scene=%s\n", ob->id.name,
                scene ? scene->id.name : "(null)");
  }
  return path_ok;
}

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
    const IDProperty *cull_cam_prop = tessera_property_find(
        const_cast<Object *>(state_ob), "tessera_cull_camera");
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

static const bNodeLink *tessera_find_link_to_socket(const bNodeTree *ntree,
                                                    const bNode *tonode,
                                                    const bNodeSocket *tosock)
{
  if (ntree == nullptr || tonode == nullptr || tosock == nullptr) {
    return nullptr;
  }
  for (const bNodeLink *link = static_cast<const bNodeLink *>(ntree->links.first);
       link != nullptr;
       link = link->next)
  {
    if (link->tonode == tonode && link->tosock == tosock) {
      return link;
    }
  }
  return nullptr;
}

static const bNodeSocket *tessera_find_node_socket(const bNode *node, int in_out, const char *name)
{
  if (node == nullptr || name == nullptr) {
    return nullptr;
  }
  const ListBase *sockets = (in_out == SOCK_IN) ? &node->inputs : &node->outputs;
  const bNodeSocket *fallback = nullptr;
  for (const bNodeSocket *sock = static_cast<const bNodeSocket *>(sockets->first);
       sock != nullptr;
       sock = sock->next)
  {
    if (STREQ(sock->name, name) || STREQ(sock->identifier, name)) {
      if (sock->flag & SOCK_IS_LINKED) {
        return sock;
      }
      if (fallback == nullptr) {
        fallback = sock;
      }
    }
  }
  return fallback;
}

static float tessera_get_socket_float(const bNodeSocket *sock, float fallback = 0.0f)
{
  if (sock != nullptr && sock->default_value != nullptr) {
    if (sock->type == SOCK_FLOAT) {
      const auto *val = static_cast<const bNodeSocketValueFloat *>(sock->default_value);
      return val->value;
    }
  }
  return fallback;
}

static float4 tessera_get_socket_rgba(const bNodeSocket *sock, const float4 &fallback)
{
  if (sock != nullptr && sock->default_value != nullptr) {
    if (sock->type == SOCK_RGBA) {
      const auto *val = static_cast<const bNodeSocketValueRGBA *>(sock->default_value);
      return float4(val->value[0], val->value[1], val->value[2], val->value[3]);
    }
  }
  return fallback;
}

static gpu::Texture *tessera_get_gpu_texture(const bNode *node)
{
  if (node == nullptr || node->type_legacy != SH_NODE_TEX_IMAGE || node->id == nullptr) {
    return nullptr;
  }
  blender::Image *ima = reinterpret_cast<blender::Image *>(node->id);
  const NodeTexImage *storage = static_cast<const NodeTexImage *>(node->storage);
  const ImageUser *iuser = storage ? &storage->iuser : nullptr;
  gpu::Texture *tex = BKE_image_get_gpu_texture(ima, const_cast<ImageUser *>(iuser));
  if (tex == nullptr) {
    ImageGPUTextures gputex = BKE_image_get_gpu_material_texture(
        ima, const_cast<ImageUser *>(iuser), false);
    tex = gputex.texture ? *gputex.texture : nullptr;
  }
  return tex;
}

struct PrincipledContext {
  const bNode *principled = nullptr;
  const bNodeTree *ntree = nullptr;
  const bNode *outer_group = nullptr;
  const bNodeTree *outer_ntree = nullptr;
};

static PrincipledContext tessera_find_principled(bNodeTree *root_ntree)
{
  PrincipledContext ctx;
  if (root_ntree == nullptr) {
    return ctx;
  }

  for (bNode *node = static_cast<bNode *>(root_ntree->nodes.first); node != nullptr;
       node = node->next)
  {
    if (node->type_legacy == SH_NODE_OUTPUT_MATERIAL) {
      const bNodeSocket *surf_sock = tessera_find_node_socket(node, SOCK_IN, "Surface");
      const bNodeLink *link = tessera_find_link_to_socket(root_ntree, node, surf_sock);
      if (link != nullptr && link->fromnode != nullptr) {
        if (link->fromnode->type_legacy == SH_NODE_BSDF_PRINCIPLED) {
          ctx.principled = link->fromnode;
          ctx.ntree = root_ntree;
          return ctx;
        }
        if (link->fromnode->type_legacy == NODE_GROUP && link->fromnode->id != nullptr) {
          bNodeTree *grp_tree = reinterpret_cast<bNodeTree *>(link->fromnode->id);
          for (bNode *gnode = static_cast<bNode *>(grp_tree->nodes.first); gnode != nullptr;
               gnode = gnode->next)
          {
            if (gnode->type_legacy == SH_NODE_BSDF_PRINCIPLED) {
              ctx.principled = gnode;
              ctx.ntree = grp_tree;
              ctx.outer_group = link->fromnode;
              ctx.outer_ntree = root_ntree;
              return ctx;
            }
          }
        }
      }
    }
  }

  for (bNode *node = static_cast<bNode *>(root_ntree->nodes.first); node != nullptr;
       node = node->next)
  {
    if (node->type_legacy == SH_NODE_BSDF_PRINCIPLED) {
      ctx.principled = node;
      ctx.ntree = root_ntree;
      return ctx;
    }
  }

  for (bNode *node = static_cast<bNode *>(root_ntree->nodes.first); node != nullptr;
       node = node->next)
  {
    if (node->type_legacy == NODE_GROUP && node->id != nullptr) {
      bNodeTree *grp_tree = reinterpret_cast<bNodeTree *>(node->id);
      for (bNode *gnode = static_cast<bNode *>(grp_tree->nodes.first); gnode != nullptr;
           gnode = gnode->next)
      {
        if (gnode->type_legacy == SH_NODE_BSDF_PRINCIPLED) {
          ctx.principled = gnode;
          ctx.ntree = grp_tree;
          ctx.outer_group = node;
          ctx.outer_ntree = root_ntree;
          return ctx;
        }
      }
    }
  }

  return ctx;
}

struct RoughnessResult {
  gpu::Texture *tex = nullptr;
  float gamma = 1.0f;
  float scale = 1.0f;
  float offset = 0.0f;
};

static RoughnessResult tessera_trace_roughness(const bNodeSocket *start_sock,
                                               const bNode *start_node,
                                               const bNodeTree *start_tree,
                                               const bNode *outer_group,
                                               const bNodeTree *outer_tree)
{
  RoughnessResult res;
  const bNodeSocket *curr_sock = start_sock;
  const bNode *curr_node = start_node;
  const bNodeTree *curr_tree = start_tree;
  const bNode *curr_outer_group = outer_group;
  const bNodeTree *curr_outer_tree = outer_tree;

  for (int depth = 0; depth < 10; ++depth) {
    const bNodeLink *link = tessera_find_link_to_socket(curr_tree, curr_node, curr_sock);
    if (link == nullptr || link->fromnode == nullptr) {
      break;
    }
    const bNode *from_node = link->fromnode;
    const bNodeSocket *from_sock = link->fromsock;

    if (from_node->type_legacy == SH_NODE_TEX_IMAGE) {
      res.tex = tessera_get_gpu_texture(from_node);
      break;
    }
    if (from_node->type_legacy == SH_NODE_GAMMA) {
      const bNodeSocket *g_sock = tessera_find_node_socket(from_node, SOCK_IN, "Gamma");
      float g_val = 1.0f;
      const bNodeLink *g_link = tessera_find_link_to_socket(curr_tree, from_node, g_sock);
      if (g_link != nullptr && g_link->fromnode != nullptr &&
          g_link->fromnode->type_legacy == NODE_GROUP_INPUT && curr_outer_group != nullptr)
      {
        const bNodeSocket *outer_sock = tessera_find_node_socket(
            curr_outer_group, SOCK_IN, g_link->fromsock->name);
        if (outer_sock == nullptr) {
          outer_sock = tessera_find_node_socket(
              curr_outer_group, SOCK_IN, g_link->fromsock->identifier);
        }
        g_val = tessera_get_socket_float(outer_sock, 1.0f);
      }
      else {
        g_val = tessera_get_socket_float(g_sock, 1.0f);
      }
      if (g_val > 0.0f) {
        res.gamma *= g_val;
      }
      curr_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color");
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == SH_NODE_MATH) {
      const bNodeSocket *val2_sock = tessera_find_node_socket(from_node, SOCK_IN, "Value_001");
      if (val2_sock == nullptr) {
        const bNodeSocket *first_sock = static_cast<const bNodeSocket *>(from_node->inputs.first);
        if (first_sock != nullptr && first_sock->next != nullptr) {
          val2_sock = first_sock->next;
        }
      }
      float op_val = tessera_get_socket_float(val2_sock, 1.0f);
      int op = from_node->custom1;
      if (op == 1) { // MULTIPLY
        res.scale *= op_val;
      }
      else if (op == 0) { // ADD
        res.offset += op_val;
      }
      else if (op == 7) { // POWER
        if (op_val > 0.0f) res.gamma *= op_val;
      }
      curr_sock = static_cast<const bNodeSocket *>(from_node->inputs.first);
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == SH_NODE_INVERT) {
      res.scale *= -1.0f;
      res.offset += 1.0f;
      curr_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color");
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == NODE_GROUP_INPUT && curr_outer_group != nullptr && curr_outer_tree != nullptr) {
      const bNodeSocket *outer_sock = tessera_find_node_socket(
          curr_outer_group, SOCK_IN, from_sock->name);
      if (outer_sock == nullptr) {
        outer_sock = tessera_find_node_socket(
            curr_outer_group, SOCK_IN, from_sock->identifier);
      }
      if (outer_sock != nullptr) {
        curr_sock = outer_sock;
        curr_node = curr_outer_group;
        curr_tree = curr_outer_tree;
        curr_outer_group = nullptr;
        curr_outer_tree = nullptr;
        continue;
      }
      break;
    }
    if (from_node->type_legacy == NODE_GROUP && from_node->id != nullptr) {
      bNodeTree *sub_tree = reinterpret_cast<bNodeTree *>(from_node->id);
      for (bNode *sub_node = static_cast<bNode *>(sub_tree->nodes.first); sub_node != nullptr;
           sub_node = sub_node->next)
      {
        if (sub_node->type_legacy == NODE_GROUP_OUTPUT) {
          const bNodeSocket *sub_sock = tessera_find_node_socket(sub_node, SOCK_IN, from_sock->name);
          if (sub_sock != nullptr) {
            curr_sock = sub_sock;
            curr_node = sub_node;
            curr_outer_group = from_node;
            curr_outer_tree = curr_tree;
            curr_tree = sub_tree;
            break;
          }
        }
      }
      continue;
    }
    break;
  }
  return res;
}

struct TextureTraceResult {
  gpu::Texture *tex = nullptr;
  bool is_srgb = true;
  float4 tint = float4(1.0f, 1.0f, 1.0f, 1.0f);
};

static TextureTraceResult tessera_trace_texture(const bNodeSocket *start_sock,
                                                const bNode *start_node,
                                                const bNodeTree *start_tree,
                                                const bNode *outer_group,
                                                const bNodeTree *outer_tree)
{
  TextureTraceResult res;
  const bNodeSocket *curr_sock = start_sock;
  const bNode *curr_node = start_node;
  const bNodeTree *curr_tree = start_tree;
  const bNode *curr_outer_group = outer_group;
  const bNodeTree *curr_outer_tree = outer_tree;

  for (int depth = 0; depth < 12; ++depth) {
    const bNodeLink *link = tessera_find_link_to_socket(curr_tree, curr_node, curr_sock);
    if (link == nullptr || link->fromnode == nullptr) {
      break;
    }
    const bNode *from_node = link->fromnode;
    const bNodeSocket *from_sock = link->fromsock;

    if (from_node->type_legacy == SH_NODE_TEX_IMAGE) {
      res.tex = tessera_get_gpu_texture(from_node);
      if (from_node->id != nullptr) {
        blender::Image *ima = reinterpret_cast<blender::Image *>(from_node->id);
        if (ima->colorspace_settings.name[0] != '\0') {
          const char *cs = ima->colorspace_settings.name;
          if (BLI_strcaseeq(cs, "Non-Color") ||
              BLI_strcaseeq(cs, "Linear") ||
              BLI_strcaseeq(cs, "Linear CIE-XYZ D65") ||
              BLI_strcaseeq(cs, "Linear Rec.709") ||
              BLI_strcaseeq(cs, "Raw")) {
            res.is_srgb = false;
          }
        }
      }
      return res;
    }
    if (from_node->type_legacy == SH_NODE_BEVEL) {
      curr_sock = tessera_find_node_socket(from_node, SOCK_IN, "Normal");
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == SH_NODE_BUMP) {
      const bNodeSocket *n_in = tessera_find_node_socket(from_node, SOCK_IN, "Normal");
      if (n_in != nullptr && (n_in->flag & SOCK_IS_LINKED)) {
        curr_sock = n_in;
      }
      else {
        curr_sock = tessera_find_node_socket(from_node, SOCK_IN, "Height");
      }
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == SH_NODE_NORMAL_MAP) {
      curr_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color");
      curr_node = from_node;
      continue;
    }
    if (from_node->type_legacy == SH_NODE_CURVE_RGB ||
        from_node->type_legacy == SH_NODE_VALTORGB ||
        from_node->type_legacy == SH_NODE_HUE_SAT) {
      const bNodeSocket *in_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color");
      if (in_sock == nullptr) in_sock = tessera_find_node_socket(from_node, SOCK_IN, "Image");
      if (in_sock == nullptr) in_sock = tessera_find_node_socket(from_node, SOCK_IN, "Fac");
      if (in_sock != nullptr) {
        curr_sock = in_sock;
        curr_node = from_node;
        continue;
      }
    }
    if (from_node->type_legacy == SH_NODE_MIX || from_node->type_legacy == SH_NODE_MIX_RGB_LEGACY) {
      const bNodeSocket *a_sock = tessera_find_node_socket(from_node, SOCK_IN, "A");
      if (a_sock == nullptr || !(a_sock->flag & SOCK_IS_LINKED)) {
        a_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color1");
      }
      if (a_sock == nullptr || !(a_sock->flag & SOCK_IS_LINKED)) {
        a_sock = tessera_find_node_socket(from_node, SOCK_IN, "A_Color");
      }
      const bNodeSocket *b_sock = tessera_find_node_socket(from_node, SOCK_IN, "B");
      if (b_sock == nullptr || !(b_sock->flag & SOCK_IS_LINKED)) {
        b_sock = tessera_find_node_socket(from_node, SOCK_IN, "Color2");
      }
      if (b_sock == nullptr || !(b_sock->flag & SOCK_IS_LINKED)) {
        b_sock = tessera_find_node_socket(from_node, SOCK_IN, "B_Color");
      }

      const bNodeSocket *follow_sock = a_sock;
      const bNodeSocket *other_sock = b_sock;
      if ((follow_sock == nullptr || !(follow_sock->flag & SOCK_IS_LINKED)) &&
          (b_sock != nullptr && (b_sock->flag & SOCK_IS_LINKED))) {
        follow_sock = b_sock;
        other_sock = a_sock;
      }

      /* Check if other branch provides a constant tint */
      if (other_sock != nullptr) {
        const bNodeSocket *fac_sock = tessera_find_node_socket(from_node, SOCK_IN, "Factor");
        if (fac_sock == nullptr) fac_sock = tessera_find_node_socket(from_node, SOCK_IN, "Fac");
        if (fac_sock == nullptr) fac_sock = tessera_find_node_socket(from_node, SOCK_IN, "Factor_Float");

        float fac = 0.0f;
        if (fac_sock != nullptr) {
          if ((fac_sock->flag & SOCK_IS_LINKED) && curr_outer_group != nullptr) {
            const bNodeLink *fac_link = tessera_find_link_to_socket(curr_tree, from_node, fac_sock);
            if (fac_link != nullptr && fac_link->fromnode != nullptr &&
                fac_link->fromnode->type_legacy == NODE_GROUP_INPUT) {
              const bNodeSocket *og_fac = tessera_find_node_socket(
                  curr_outer_group, SOCK_IN, fac_link->fromsock->name);
              if (og_fac == nullptr) {
                og_fac = tessera_find_node_socket(
                    curr_outer_group, SOCK_IN, fac_link->fromsock->identifier);
              }
              fac = tessera_get_socket_float(og_fac, 0.0f);
            }
          }
          else {
            fac = tessera_get_socket_float(fac_sock, 0.0f);
          }
        }

        if (fac > 1e-4f && !(other_sock->flag & SOCK_IS_LINKED)) {
          float4 other_col = tessera_get_socket_rgba(other_sock, float4(1.0f));
          int blend_type = 0; // MA_RAMP_BLEND
          if (from_node->type_legacy == SH_NODE_MIX && from_node->storage != nullptr) {
            const NodeShaderMix *mix_storage = static_cast<const NodeShaderMix *>(from_node->storage);
            blend_type = mix_storage->blend_type;
          }
          else if (from_node->type_legacy == SH_NODE_MIX_RGB_LEGACY) {
            blend_type = from_node->custom1;
          }
          ramp_blend(blend_type, &res.tint[0], fac, &other_col[0]);
        }
      }

      if (follow_sock != nullptr) {
        curr_sock = follow_sock;
        curr_node = from_node;
        continue;
      }
    }
    if (from_node->type_legacy == NODE_GROUP_INPUT && curr_outer_group != nullptr && curr_outer_tree != nullptr) {
      const bNodeSocket *outer_sock = tessera_find_node_socket(
          curr_outer_group, SOCK_IN, from_sock->name);
      if (outer_sock == nullptr) {
        outer_sock = tessera_find_node_socket(
            curr_outer_group, SOCK_IN, from_sock->identifier);
      }
      if (outer_sock != nullptr) {
        curr_sock = outer_sock;
        curr_node = curr_outer_group;
        curr_tree = curr_outer_tree;
        curr_outer_group = nullptr;
        curr_outer_tree = nullptr;
        continue;
      }
      break;
    }
    break;
  }
  return res;
}

static std::vector<tessera::TesseraMaterialTextures> tessera_extract_materials(Object *ob)
{
  std::vector<tessera::TesseraMaterialTextures> result;
  if (ob == nullptr) {
    return result;
  }
  const int totcol = std::max(1, int(ob->totcol));
  result.resize(totcol);

  for (int i = 0; i < totcol; ++i) {
    auto *ma = BKE_object_material_get(ob, i + 1);
    tessera::TesseraMaterialTextures &mat = result[i];
    if (ma == nullptr) {
      continue;
    }
    mat.base_color = float4(ma->r, ma->g, ma->b, ma->a);
    mat.roughness_value = ma->roughness;
    mat.metallic_value = ma->metallic;
    mat.specular_value = (ma->spec > 0.0f) ? ma->spec : 0.5f;

    if (!ma->use_nodes || ma->nodetree == nullptr) {
      continue;
    }

    PrincipledContext pctx = tessera_find_principled(ma->nodetree);
    if (pctx.principled != nullptr) {
      const bNode *pnode = pctx.principled;
      const bNodeTree *ntree = pctx.ntree;
      const bNode *ogrp = pctx.outer_group;
      const bNodeTree *ontree = pctx.outer_ntree;

      const bNodeSocket *r_sock = tessera_find_node_socket(pnode, SOCK_IN, "Roughness");
      if (r_sock != nullptr) {
        if (r_sock->flag & SOCK_IS_LINKED) {
          RoughnessResult rr = tessera_trace_roughness(r_sock, pnode, ntree, ogrp, ontree);
          if (rr.tex != nullptr) {
            mat.roughness = rr.tex;
            mat.has_roughness = true;
            mat.roughness_gamma = rr.gamma;
            mat.roughness_scale = rr.scale;
            mat.roughness_offset = rr.offset;
          }
        }
        else {
          mat.roughness_value = tessera_get_socket_float(r_sock, mat.roughness_value);
          mat.has_roughness = false;
        }
      }

      const bNodeSocket *c_sock = tessera_find_node_socket(pnode, SOCK_IN, "Base Color");
      if (c_sock != nullptr) {
        if (c_sock->flag & SOCK_IS_LINKED) {
          TextureTraceResult tr = tessera_trace_texture(c_sock, pnode, ntree, ogrp, ontree);
          if (tr.tex != nullptr) {
            mat.diffuse = tr.tex;
            mat.has_diffuse = true;
            mat.diffuse_is_srgb = tr.is_srgb;
            mat.base_color = tr.tint;
          }
        }
        else {
          mat.base_color = tessera_get_socket_rgba(c_sock, mat.base_color);
        }
      }

      const bNodeSocket *n_sock = tessera_find_node_socket(pnode, SOCK_IN, "Normal");
      if (n_sock != nullptr && (n_sock->flag & SOCK_IS_LINKED)) {
        TextureTraceResult tr = tessera_trace_texture(n_sock, pnode, ntree, ogrp, ontree);
        if (tr.tex != nullptr) {
          mat.normal = tr.tex;
          mat.has_normal = true;
        }
      }

      const bNodeSocket *m_sock = tessera_find_node_socket(pnode, SOCK_IN, "Metallic");
      if (m_sock != nullptr && !(m_sock->flag & SOCK_IS_LINKED)) {
        mat.metallic_value = tessera_get_socket_float(m_sock, mat.metallic_value);
      }

      const bNodeSocket *s_sock = tessera_find_node_socket(pnode, SOCK_IN, "Specular IOR Level");
      if (s_sock == nullptr) {
        s_sock = tessera_find_node_socket(pnode, SOCK_IN, "Specular");
      }
      if (s_sock != nullptr) {
        if (s_sock->flag & SOCK_IS_LINKED) {
          const bNodeLink *slink = tessera_find_link_to_socket(ntree, pnode, s_sock);
          if (slink != nullptr && slink->fromnode != nullptr) {
            const bNode *s_from = slink->fromnode;
            if (s_from->type_legacy == SH_NODE_GAMMA) {
              /* Inverted roughness gamma curve on gravel/rock gives low dielectric specular ~0.08 */
              mat.specular_value = 0.08f;
            }
            else {
              mat.specular_value = 0.5f;
            }
          }
          else {
            mat.specular_value = 0.5f;
          }
        }
        else {
          mat.specular_value = tessera_get_socket_float(s_sock, 0.5f);
        }
      }

      const bNodeSocket *st_sock = tessera_find_node_socket(pnode, SOCK_IN, "Specular Tint");
      if (st_sock != nullptr && (st_sock->flag & SOCK_IS_LINKED)) {
        mat.has_specular_tint = true;
      }
    }

    auto search_tex_nodes = [&](bNodeTree *tree) {
      if (tree == nullptr) return;
      for (bNode *node = static_cast<bNode *>(tree->nodes.first); node != nullptr; node = node->next) {
        if (node->type_legacy != SH_NODE_TEX_IMAGE || node->id == nullptr) continue;
        gpu::Texture *tex = tessera_get_gpu_texture(node);
        if (tex == nullptr) continue;
        std::string node_name = node->name;
        std::string ima_name = (reinterpret_cast<blender::Image *>(node->id)->id.name + 2);
        for (char &c : node_name) c = std::tolower(c);
        for (char &c : ima_name) c = std::tolower(c);

        bool is_normal = (node_name.find("normal") != std::string::npos ||
                          node_name.find("norm") != std::string::npos ||
                          ima_name.find("normal") != std::string::npos ||
                          ima_name.find("norm") != std::string::npos);

        bool is_rough = (node_name.find("rough") != std::string::npos ||
                         node_name.find("rgh") != std::string::npos ||
                         ima_name.find("rough") != std::string::npos ||
                         ima_name.find("rgh") != std::string::npos);

        bool is_ao = (node_name.find("ao") != std::string::npos ||
                      node_name.find("occl") != std::string::npos ||
                      ima_name.find("ao") != std::string::npos ||
                      ima_name.find("occl") != std::string::npos);

        bool is_diff = (node_name.find("diff") != std::string::npos ||
                        node_name.find("base") != std::string::npos ||
                        node_name.find("color") != std::string::npos ||
                        node_name.find("albedo") != std::string::npos ||
                        ima_name.find("diff") != std::string::npos ||
                        ima_name.find("base") != std::string::npos ||
                        ima_name.find("color") != std::string::npos ||
                        ima_name.find("albedo") != std::string::npos);

        if (is_normal && !mat.has_normal) {
          mat.normal = tex;
          mat.has_normal = true;
        }
        else if (is_rough && !mat.has_roughness) {
          mat.roughness = tex;
          mat.has_roughness = true;
        }
        else if (is_diff && !mat.has_diffuse) {
          mat.diffuse = tex;
          mat.has_diffuse = true;
          blender::Image *ima = reinterpret_cast<blender::Image *>(node->id);
          if (ima != nullptr && ima->colorspace_settings.name[0] != '\0') {
            const char *cs = ima->colorspace_settings.name;
            if (BLI_strcaseeq(cs, "Non-Color") ||
                BLI_strcaseeq(cs, "Linear") ||
                BLI_strcaseeq(cs, "Linear CIE-XYZ D65") ||
                BLI_strcaseeq(cs, "Linear Rec.709") ||
                BLI_strcaseeq(cs, "Raw")) {
              mat.diffuse_is_srgb = false;
            }
          }
        }
        else if (!mat.has_diffuse && !is_normal && !is_rough && !is_ao) {
          mat.diffuse = tex;
          mat.has_diffuse = true;
          blender::Image *ima = reinterpret_cast<blender::Image *>(node->id);
          if (ima != nullptr && ima->colorspace_settings.name[0] != '\0') {
            const char *cs = ima->colorspace_settings.name;
            if (BLI_strcaseeq(cs, "Non-Color") ||
                BLI_strcaseeq(cs, "Linear") ||
                BLI_strcaseeq(cs, "Linear CIE-XYZ D65") ||
                BLI_strcaseeq(cs, "Linear Rec.709") ||
                BLI_strcaseeq(cs, "Raw")) {
              mat.diffuse_is_srgb = false;
            }
          }
        }
      }
    };

    if (!mat.has_diffuse || !mat.has_normal || !mat.has_roughness) {
      search_tex_nodes(ma->nodetree);
      for (bNode *node = static_cast<bNode *>(ma->nodetree->nodes.first); node != nullptr; node = node->next) {
        if (node->type_legacy == NODE_GROUP && node->id != nullptr) {
          search_tex_nodes(reinterpret_cast<bNodeTree *>(node->id));
        }
      }
    }
  }

  for (auto &mat : result) {
    if (mat.diffuse != nullptr) {
      GPU_texture_mipmap_mode(mat.diffuse, true, true);
    }
    if (mat.normal != nullptr) {
      GPU_texture_mipmap_mode(mat.normal, true, true);
    }
    if (mat.roughness != nullptr) {
      GPU_texture_mipmap_mode(mat.roughness, true, true);
    }
  }
  return result;
}

static std::string tessera_texture_lod_mode_get(Object *ob)
{
  if (ob == nullptr) {
    return "AUTO";
  }
  const IDProperty *prop = tessera_property_find(ob, "tessera_texture_lod_mode");
  if (prop != nullptr && prop->type == IDP_STRING) {
    const char *str = IDP_string_get(prop);
    if (str != nullptr && str[0] != '\0') {
      return std::string(str);
    }
  }
  return "AUTO";
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

bool TesseraModule::is_enabled(const Scene *scene, Object *ob) const
{
  if (!is_tessera_eligible_mesh(ob)) {
    if (tess_eevee_dbg_enabled() && ob != nullptr && ob->data != nullptr) {
      tess_eevee_log("[tess-eevee] is_enabled NO ineligible: name=%s type=%d\n",
                  (ob->id.name) ? (ob->id.name + 2) : "?", int(ob->type));
    }
    return false;
  }
  if (!tessera_enabled_get(scene, ob)) {
    if (tess_eevee_dbg_enabled()) {
      tess_eevee_log("[tess-eevee] is_enabled NO enabled-get: name=%s\n",
                  (ob->id.name) ? (ob->id.name + 2) : "?");
    }
    return false;
  }
  std::string dummy;
  const bool path_ok = tessera_path_get(ob, dummy);
  if (tess_eevee_dbg_enabled()) {
    tess_eevee_log("[tess-eevee] is_enabled: name=%s path_ok=%d\n",
                (ob->id.name) ? (ob->id.name + 2) : "?", int(path_ok));
  }
  return path_ok;
}

bool TesseraModule::record(const draw::ObjectRef &ob_ref)
{
  Object *ob = ob_ref.object;
  if (ob == nullptr) {
    return false;
  }

  const bool is_dupli = (ob->base_flag & BASE_FROM_DUPLI) != 0;
  const bool is_range = ob_ref.is_range();

  if (tess_eevee_dbg_enabled()) {
    tess_eevee_log("[tess-eevee] record-enter: name=%s dupli=%d range=%d\n",
                (ob->id.name) ? (ob->id.name + 2) : "?",
                int(is_dupli),
                int(is_range));
  }

  /* Fast path for consecutive instances of the same object / archetype in scatter fields.
   * Bypasses IDProperty lookups, path conversions, archetype fingerprinting and hash lookups. */
  if (tessera_last_state_ != nullptr &&
      tessera_last_ob_ == ob &&
      tessera_last_data_ == ob->data &&
      tessera_last_sync_round_ == this->tessera_sync_round_)
  {
    if (tess_eevee_dbg_enabled()) {
      tess_eevee_log("[tess-eevee] record-fastpath: inst_mats=%zu\n",
                  tessera_last_state_->instance_mats.size());
    }
    if (is_range) {
      const int range_count = ob_ref.instances_count();
      const size_t room = (tessera_last_state_->instance_mats.size() < 65536) ?
                              (65536 - tessera_last_state_->instance_mats.size()) :
                              0;
      const int to_copy = std::min(int(room), range_count);
      const size_t old_sz = tessera_last_state_->instance_mats.size();
      tessera_last_state_->instance_mats.resize(old_sz + to_copy);
      for (int i = 0; i < to_copy; ++i) {
        tessera_last_state_->instance_mats[old_sz + i] = ob_ref.object_to_world(i);
      }
    }
    else if (is_dupli) {
      if (tessera_last_state_->instance_mats.size() < 65536) {
        tessera_last_state_->instance_mats.push_back(ob->object_to_world());
      }
    }
    return true;
  }

  if (!is_tessera_eligible_mesh(ob)) {
    if (tess_eevee_dbg_enabled()) {
      tess_eevee_log("[tess-eevee] record-bail: ineligible type=%d data=%p\n", int(ob->type), ob->data);
    }
    return false;
  }
  if (!tessera_enabled_get(inst_.scene, ob)) {
    if (tess_eevee_dbg_enabled()) {
      tess_eevee_log("[tess-eevee] record-bail: tessera_enabled_get false\n");
    }
    return false;
  }
  std::string idprop_path;
  if (!tessera_path_get(ob, idprop_path)) {
    if (tess_eevee_dbg_enabled()) {
      tess_eevee_log("[tess-eevee] record-bail: tessera_path_get false\n");
    }
    return false;
  }
  char abs_path[FILE_MAX];
  BLI_strncpy(abs_path, idprop_path.c_str(), sizeof(abs_path));
  const char *blend_path = BKE_main_blendfile_path_from_global();
  if (blend_path != nullptr && blend_path[0] != '\0') {
    BLI_path_abs(abs_path, blend_path);
  }

  /* Resolve the realized-geometry archetype for this dupli range so EEVEE
   * draws the actual plant mesh (not the low-poly emitter proxy) with the
   * correct material color. */
  tessera::TesseraArchetype arch;
  bool arch_resolved = false;
  Main *arch_bmain = (inst_.draw_ctx != nullptr && inst_.draw_ctx->depsgraph != nullptr) ?
                         DEG_get_bmain(inst_.draw_ctx->depsgraph) :
                         nullptr;
  if (is_dupli || is_range) {
    if (const ID *range_data = static_cast<const ID *>(ob->data)) {
      arch = tessera::ArchetypeResolver::global().resolve(
          arch_bmain, blend_path, ob, range_data);
      arch_resolved = arch.valid;
    }
  }

  /* Build state key from emitter name + archetype so different plant
   * varieties of the same scatter field get separate .nc2 data. */
  std::string state_key(ob->id.name + 2);
  if (arch_resolved && !arch.source_name.empty()) {
    state_key += "|";
    state_key += arch.source_name;
  }

  std::lock_guard lock(this->mutex_);
  std::unique_ptr<ObjectState> &entry = objects_.lookup_or_add_default(state_key);
  if (entry == nullptr) {
    entry = std::make_unique<ObjectState>();
  }
  ObjectState &state = *entry;

  tessera_last_state_ = &state;
  tessera_last_ob_ = ob;
  tessera_last_data_ = ob->data;
  tessera_last_sync_round_ = this->tessera_sync_round_;

  if (tess_eevee_dbg_enabled()) {
    tess_eevee_log("[tess-eevee] record: key=%s dupli=%d range=%d inst_mats=%zu clusters=%zu used=%d\n",
                state_key.c_str(),
                int(is_dupli),
                int(is_range),
                state.instance_mats.size(),
                state.data.clusters.size(),
                int(state.used));
  }

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
    state.show_stats = tessera_show_stats_get(ob);
    state.log_console = tessera_log_console_get(ob);
    state.show_fps_panel = tessera_show_fps_panel_get(ob);
    state.show_fps_hud = tessera_show_fps_hud_get(ob);
    state.visualize_clusters = tessera_visualize_get(ob);
    state.pixel_error = tessera_pixel_error_get(ob);
    state.view_source = tessera_view_source_get(ob);
    state.culling_margin = tessera_culling_margin_get(ob);
    state.object_mat = ob_ref.is_range() ? ob_ref.object_to_world(0) : ob->object_to_world();
    /* Respect the viewport shading backface-culling flag: native Blender treats
     * objects as two-sided (back faces rasterize with flipped normals) unless
     * the user explicitly enables Backface Culling in the Viewport Shading
     * popover (v3d->shading.flag & V3D_SHADING_BACKFACE_CULLING). */
    bool cull_backfaces = (inst_.v3d != nullptr) && (inst_.v3d->shading.flag & V3D_SHADING_BACKFACE_CULLING);
    state.two_sided = !cull_backfaces || tessera_object_is_two_sided(ob) || (arch_resolved && arch.two_sided);
    state.is_instancer_base = false;
    if (arch_resolved && arch.base_color_valid) {
      state.base_color = arch.base_color;
      state.base_color_valid = true;
    }
  }

  if (!is_dupli && !is_range) {
    const bool is_instancer = (ob->transflag & OB_DUPLI) != 0 ||
                              (ob->runtime && ob->runtime->geometry_set_eval &&
                               ob->runtime->geometry_set_eval->has_instances());
    if (is_instancer) {
      state.is_instancer_base = true;
      /* Preserve all child archetype states cached for this instancer across frames */
      const std::string prefix = std::string(ob->id.name + 2) + "|";
      for (auto item : objects_.items()) {
        if (item.key.rfind(prefix, 0) == 0) {
          item.value->used = true;
        }
      }
    }
  }

  if (ob_ref.is_range()) {
    const int range_count = ob_ref.instances_count();
    const size_t room = (state.instance_mats.size() < 65536) ?
                            (65536 - state.instance_mats.size()) :
                            0;
    const int to_copy = std::min(int(room), range_count);
    const size_t old_sz = state.instance_mats.size();
    state.instance_mats.resize(old_sz + to_copy);
    for (int i = 0; i < to_copy; ++i) {
      state.instance_mats[old_sz + i] = ob_ref.object_to_world(i);
    }
  }
  else if (is_dupli) {
    if (state.instance_mats.size() < 65536) {
      state.instance_mats.push_back(ob->object_to_world());
    }
  }

  /* Material extraction caching: re-extract only when materials or images are updated. */
  Object *mat_source_ob = (arch_resolved && !arch.matched_object_name.empty() && arch_bmain != nullptr) ?
                              static_cast<Object *>(BLI_findstring(&arch_bmain->objects,
                                                                   arch.matched_object_name.c_str(),
                                                                   offsetof(ID, name) + 2)) :
                              nullptr;
  if (mat_source_ob == nullptr) {
    mat_source_ob = ob;
  }
  const int totcol = std::max(1, int(mat_source_ob->totcol));
  bool materials_dirty = !state.materials_extracted || (state.cached_totcol != totcol);
  if (!materials_dirty && state.cached_materials.size() == size_t(totcol)) {
    for (int i = 0; i < totcol; ++i) {
      void *ma = static_cast<void *>(BKE_object_material_get(mat_source_ob, i + 1));
      if (ma != state.cached_materials[i]) {
        materials_dirty = true;
        break;
      }
    }
  }
  else {
    materials_dirty = true;
  }

  if (!materials_dirty && inst_.draw_ctx && inst_.draw_ctx->depsgraph) {
    if (DEG_id_type_updated(inst_.draw_ctx->depsgraph, ID_MA) ||
        DEG_id_type_updated(inst_.draw_ctx->depsgraph, ID_IM) ||
        DEG_id_type_updated(inst_.draw_ctx->depsgraph, ID_NT))
    {
      materials_dirty = true;
    }
  }

  if (materials_dirty) {
    state.materials = tessera_extract_materials(mat_source_ob);
    state.cached_totcol = totcol;
    state.cached_materials.resize(totcol);
    for (int i = 0; i < totcol; ++i) {
      state.cached_materials[i] = static_cast<void *>(BKE_object_material_get(mat_source_ob, i + 1));
    }
    state.materials_extracted = true;
  }
  if (state.base_color_valid) {
    /* Propagate the archetype's material base color into the first material
     * so the GPU shader uses the correct foliage color. ScatterFlow emitter
     * proxies carry no material slots — the color lives in the archetype. */
    if (totcol >= 1 && state.materials.size() >= 1) {
      state.materials[0].base_color = state.base_color;
    }
  }

  if (arch_resolved && arch.built_data != nullptr) {
    if (state.data_source_key != arch.source_name || state.data.clusters.empty()) {
      state.data = *arch.built_data;
      state.pass.reset();
    }
    state.data_source_key = arch.source_name;
    state.nc2_path = "";
    state.file_mtime = 0;
    state.file_size = 0;
  }
  else {
    const std::string load_path = (arch_resolved && !arch.nc2_path.empty()) ?
                                      arch.nc2_path :
                                      std::string(abs_path);
    BLI_stat_t load_info;
    const bool stat_ok = (BLI_stat(load_path.c_str(), &load_info) == 0);
    const int64_t load_mtime = stat_ok ? int64_t(load_info.st_mtime) : -1;
    const int64_t load_size = stat_ok ? int64_t(load_info.st_size) : -1;
    const bool stale = (state.nc2_path != load_path) || (state.file_mtime != load_mtime) ||
                       (state.file_size != load_size);
    if (state.pass == nullptr || stale) {
      if (stale || state.data.clusters.empty()) {
        size_t data_size = 0;
        if (void *file_data = BLI_file_read_binary_as_mem(load_path.c_str(), 0, &data_size)) {
          tessera::TesseraData loaded;
          if (tessera::deserializeFromMemory(
                  loaded, static_cast<const uint8_t *>(file_data), data_size) &&
              !loaded.clusters.empty())
          {
            state.data = std::move(loaded);
          }
          MEM_delete_void(file_data);
        }
        state.file_mtime = load_mtime;
        state.file_size = load_size;
        state.nc2_path = load_path;
        state.data_source_key = "";
        state.pass.reset();
      }
    }
  }
  return !state.data.clusters.empty();
}

void TesseraModule::update_telemetry(ObjectState &state, const float3 &cam_pos)
{
  const bool show_stats = state.show_stats;
  const bool log_console = state.log_console;
  const bool show_fps_panel = state.show_fps_panel;
  const bool show_fps_hud = state.show_fps_hud;

  bool is_navigating = (inst_.rv3d != nullptr) &&
                       (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || (inst_.rv3d->viewlock != 0));
  if (inst_.scene != nullptr) {
    if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
      if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
        if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
          is_navigating = true;
        }
      }
    }
  }

  bool allow_dna_write = false;
  if (!is_navigating && inst_.rv3d == nullptr) {
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

  if (!show_stats && !log_console && !show_fps_panel && !show_fps_hud) {
    state.last_frame_time = {};
    state.smoothed_dt = 0.0f;
    return;
  }

  /* Compute FPS / frame time metrics if requested. */
  if (show_stats || show_fps_panel || show_fps_hud) {
    auto now = std::chrono::steady_clock::now();
    if (state.last_frame_time.time_since_epoch().count() == 0) {
      state.last_frame_time = now;
    }
    else {
      float dt = std::chrono::duration<float>(now - state.last_frame_time).count();
      if (dt > 0.001f && dt < 1.0f) {
        state.last_frame_time = now;
        state.smoothed_dt = (state.smoothed_dt <= 0.0f) ? dt :
                                                          (0.9f * state.smoothed_dt + 0.1f * dt);
      }
      else if (dt >= 1.0f) {
        state.last_frame_time = now;
      }
    }
    float fps = (state.smoothed_dt > 1e-5f) ? (1.0f / state.smoothed_dt) : 0.0f;
    float frame_time_ms = state.smoothed_dt * 1000.0f;
    if (allow_dna_write && state.ob != nullptr) {
      tessera_write_fps_to_object(state.ob, fps, frame_time_ms);
      if (show_fps_panel) {
        WM_main_add_notifier(NC_OBJECT | ND_DRAW, state.ob);
      }
    }
  }
  else {
    state.last_frame_time = {};
    state.smoothed_dt = 0.0f;
  }

  const float3 bounds_min(state.data.header.boundsMin[0],
                          state.data.header.boundsMin[1],
                          state.data.header.boundsMin[2]);
  const float3 bounds_max(state.data.header.boundsMax[0],
                          state.data.header.boundsMax[1],
                          state.data.header.boundsMax[2]);
  const float3 local_center = 0.5f * (bounds_min + bounds_max);
  const float3 world_center = (state.object_mat * float4(local_center, 1.0f)).xyz();
  const float cam_distance = math::distance(cam_pos, world_center);

  std::string lod_mode = tessera_texture_lod_mode_get(state.ob);
  float scale = 1.0f;
  if (lod_mode == "FULL") {
    scale = 1.0f;
  }
  else if (lod_mode == "HALF") {
    scale = 0.5f;
  }
  else if (lod_mode == "QUARTER") {
    scale = 0.25f;
  }
  else if (lod_mode == "EIGHTH") {
    scale = 0.125f;
  }
  else { // "AUTO"
    const float ref_dist = 6.0f;
    const float raw_factor = std::clamp(ref_dist / std::max(cam_distance, 0.01f), 0.125f, 1.0f);
    if (raw_factor >= 0.75f) {
      scale = 1.0f;
    }
    else if (raw_factor >= 0.375f) {
      scale = 0.5f;
    }
    else if (raw_factor >= 0.1875f) {
      scale = 0.25f;
    }
    else {
      scale = 0.125f;
    }
  }
  state.current_tex_scale = scale;

  int base_w = 0, base_h = 0;
  for (const auto &mat : state.materials) {
    if (mat.diffuse != nullptr) {
      base_w = std::max(base_w, GPU_texture_width(mat.diffuse));
      base_h = std::max(base_h, GPU_texture_height(mat.diffuse));
    }
    if (mat.normal != nullptr) {
      base_w = std::max(base_w, GPU_texture_width(mat.normal));
      base_h = std::max(base_h, GPU_texture_height(mat.normal));
    }
    if (mat.roughness != nullptr) {
      base_w = std::max(base_w, GPU_texture_width(mat.roughness));
      base_h = std::max(base_h, GPU_texture_height(mat.roughness));
    }
  }
  if (base_w == 0 || base_h == 0) {
    base_w = 4096;
    base_h = 4096;
  }
  const int active_res_x = int(base_w * scale);
  const int active_res_y = int(base_h * scale);

  if (allow_dna_write && state.ob != nullptr) {
    auto write_tex_props = [&](ID *id) {
      if (id == nullptr) return;
      if (IDProperty *group = IDP_EnsureProperties(id)) {
        idp_float_replace(group, "tessera_stat_tex_scale", scale);
        idp_int_replace(group, "tessera_stat_tex_res_x", active_res_x);
        idp_int_replace(group, "tessera_stat_tex_res_y", active_res_y);
      }
    };
    write_tex_props(&state.ob->id);
    const ID *orig_id = DEG_get_original_id(&state.ob->id);
    if (orig_id != nullptr && orig_id != &state.ob->id) {
      write_tex_props(const_cast<ID *>(orig_id));
    }
  }

  /* Extract telemetry counters and update object properties (skipped completely while navigating). */
  if (!is_navigating && (show_stats || log_console || show_fps_hud)) {
    if (allow_dna_write || log_console) {
      state.last_stat_time = std::chrono::steady_clock::now();

      tessera::TesseraTelemetry telem = state.pass ? state.pass->getTelemetry() :
                                                     tessera::TesseraTelemetry{};
      telem.cam_distance = cam_distance;
      telem.pixel_error = state.pass ? state.pass->getPixelError() : state.pixel_error;
      telem.tex_scale = scale;
      telem.tex_res_x = active_res_x;
      telem.tex_res_y = active_res_y;

      tessera::TesseraMemoryBudget budget = state.pass ? state.pass->getMemoryBudget() :
                                                         tessera::TesseraMemoryBudget{};
      int vram_budget_mb = 1024;
      if (inst_.scene != nullptr) {
        if (IDProperty *group = IDP_GetProperties(&inst_.scene->id)) {
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
      uint32_t total_instances = uint32_t(objects_.size());
      uint32_t active_instances = 0;
      for (const auto it : objects_.items()) {
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
        if (show_fps_panel) {
          WM_main_add_notifier(NC_OBJECT | ND_DRAW, state.ob);
        }
        if (inst_.scene != nullptr) {
          if (IDProperty *sgroup = IDP_EnsureProperties(&inst_.scene->id)) {
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
        printf("[Tessera Stats] %s | Dist: %.2fm | LOD: %u-%u | Clusters: %u/%u | Tris: %u/%u | Error: %.1fpx | Tex: %.0f%% (%dx%d)\n",
               state.ob ? (state.ob->id.name + 2) : "Object",
               telem.cam_distance,
               telem.active_lod_min,
               telem.active_lod_max,
               telem.active_clusters,
               telem.total_clusters,
               telem.active_triangles,
               telem.total_triangles,
               telem.pixel_error,
               scale * 100.0f,
               active_res_x,
               active_res_y);
        fflush(stdout);
      }
    }
  }
}

void TesseraModule::draw_prepass(const draw::View &view,
                                 const int2 &resolution,
                                 gpu::FrameBuffer *depth_fb)
{
  if (objects_.is_empty()) {
    return;
  }
  std::lock_guard lock(this->mutex_);
  const float4x4 viewport_persmat = view.persmat();
  const float viewport_proj_scale_y = std::abs(view.winmat()[1][1]);
  /* World-space frustum planes for per-instance camera culling. */
  const tessera::TesseraViewInfo tessera_cull_view = tessera::getViewInfo(
      viewport_persmat, float(resolution.y), viewport_proj_scale_y);

  if (depth_fb != nullptr) {
    GPU_framebuffer_bind(depth_fb);
  }

  /* Phase A: Terrain and single meshes first (writes background/ground depth) */
  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty() || state.is_instancer_base || !state.instance_mats.empty()) {
      continue;
    }
    Object *ob = (inst_.draw_ctx != nullptr) ?
                     tessera_object_revalidate(inst_.draw_ctx, state.ob_name) :
                     nullptr;
    if (ob == nullptr && !state.matched_archetype_name.empty() && inst_.draw_ctx != nullptr) {
      ob = tessera_object_revalidate(inst_.draw_ctx, state.matched_archetype_name);
    }
    if (ob == nullptr) {
      if (inst_.draw_ctx != nullptr) {
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
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    const int view_source = (state.ob != nullptr) ?
        tessera_view_source_get(state.ob) : state.view_source;
    state.view_source = view_source;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    const bool is_cam_view = (state.view_source != 0) ||
                             (inst_.rv3d != nullptr && inst_.rv3d->persp == RV3D_CAMOB);
    bool has_cam_lod = false;
    if (is_cam_view) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y))
      {
        has_cam_lod = true;
      }
    }
    const float4x4 lod_object_persmat = has_cam_lod ? (cam_persmat * state.object_mat) :
                                                      object_persmat;
    const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;

    if (state.pass == nullptr) {
      state.pass = std::make_unique<tessera::TesseraDrawPass>(
          state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
      if (!state.pass->init()) {
        state.pass.reset();
        continue;
      }
    }
    state.pass->setReverseZ(true);
    const float culling_margin = (state.ob != nullptr) ?
        tessera_culling_margin_get(state.ob) : state.culling_margin;
    state.culling_margin = culling_margin;
    state.pass->setCullingMargin(culling_margin);
    const float pixel_error = (state.ob != nullptr) ?
        tessera_pixel_error_get(state.ob) : state.pixel_error;
    state.pixel_error = pixel_error;
    bool is_navigating = (inst_.rv3d != nullptr) && (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || inst_.rv3d->viewlock != 0);
    if (inst_.scene != nullptr) {
      if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
        if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
          if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
            is_navigating = true;
          }
        }
      }
    }
    if (!is_navigating && inst_.rv3d == nullptr) {
      tessera_stat_culling_margin_write(state.ob, culling_margin);
    }
    float effective_pixel_error = pixel_error;
    if (is_navigating) {
      effective_pixel_error *= 2.5f;
    }
    state.pass->setPixelError(effective_pixel_error);
    state.pass->setNavigating(is_navigating);
    state.pass->setTelemetryEnabled(state.show_stats || state.log_console || state.show_fps_panel || state.show_fps_hud);

    state.pass->update(state.data,
                       object_persmat,
                       viewport_proj_scale_y,
                       float(resolution.y),
                       has_cam_lod ? &lod_object_persmat : nullptr,
                       has_cam_lod ? &lod_proj_scale_y : nullptr);
    state.traversed_this_frame = true;

    state.pass->setTextureScale(state.current_tex_scale);
    state.pass->set_material_textures(state.materials);

    if (state.pass->drawCount() != 0) {
      state.pass->draw_prepass(resolution.x, resolution.y);
    }
  }

  /* Phase B: Scattered instance fields second (writes foliage depth with Reverse-Z) */
  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty() || state.is_instancer_base || state.instance_mats.empty()) {
      continue;
    }
    Object *ob = (inst_.draw_ctx != nullptr) ?
                     tessera_object_revalidate(inst_.draw_ctx, state.ob_name) :
                     nullptr;
    if (ob == nullptr && !state.matched_archetype_name.empty() && inst_.draw_ctx != nullptr) {
      ob = tessera_object_revalidate(inst_.draw_ctx, state.matched_archetype_name);
    }
    if (ob == nullptr) {
      if (inst_.draw_ctx != nullptr) {
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
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    const int view_source = (state.ob != nullptr) ?
        tessera_view_source_get(state.ob) : state.view_source;
    state.view_source = view_source;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    const bool is_cam_view = (state.view_source != 0) ||
                             (inst_.rv3d != nullptr && inst_.rv3d->persp == RV3D_CAMOB);
    bool has_cam_lod = false;
    if (is_cam_view) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y))
      {
        has_cam_lod = true;
      }
    }

    if (state.pass == nullptr) {
      state.pass = std::make_unique<tessera::TesseraDrawPass>(
          state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
      if (!state.pass->init()) {
        state.pass.reset();
        continue;
      }
    }
    state.pass->setReverseZ(true);
    const float culling_margin = (state.ob != nullptr) ?
        tessera_culling_margin_get(state.ob) : state.culling_margin;
    state.culling_margin = culling_margin;
    state.pass->setCullingMargin(culling_margin);
    const float pixel_error = (state.ob != nullptr) ?
        tessera_pixel_error_get(state.ob) : state.pixel_error;
    state.pixel_error = pixel_error;
    bool is_navigating = (inst_.rv3d != nullptr) && (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || inst_.rv3d->viewlock != 0);
    if (inst_.scene != nullptr) {
      if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
        if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
          if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
            is_navigating = true;
          }
        }
      }
    }
    if (!is_navigating && inst_.rv3d == nullptr) {
      tessera_stat_culling_margin_write(state.ob, culling_margin);
    }
    float effective_pixel_error = pixel_error;
    if (is_navigating) {
      effective_pixel_error *= 2.5f;
    }
    state.pass->setPixelError(effective_pixel_error);
    state.pass->setNavigating(is_navigating);
    state.pass->setTelemetryEnabled(state.show_stats || state.log_console || state.show_fps_panel || state.show_fps_hud);

    state.pass->setTextureScale(state.current_tex_scale);
    state.pass->set_material_textures(state.materials);
    state.pass->set_two_sided(state.two_sided);

    bool has_diffuse_tex = false;
    for (const auto &m : state.materials) {
      if (m.has_diffuse && m.diffuse != nullptr) {
        has_diffuse_tex = true;
        break;
      }
    }
    state.pass->set_alpha_mask(has_diffuse_tex);
    /* EEVEE uses bucketed multi-draw indirect args (like Workbench), so the
     * counter buffer must be populated with per-bucket
     * DrawArraysIndirectCommand descriptors. */
    state.pass->setUseElementsIndirect(true);

    /* Mesh bounding sphere (object space) for per-instance culling. */
    float3 tessera_bs_center(0.0f);
    float tessera_bs_radius = 0.0f;
    const bool tessera_have_bs = tessera::mesh_bounds_sphere(
        state.data, tessera_bs_center, tessera_bs_radius);

    state.frame_visible_instances.clear();
    const float max_draw_dist = eevee_max_draw_dist_get(state.ob_name, state.matched_archetype_name);
    const float max_d_sq = max_draw_dist > 0.0f ? (max_draw_dist * max_draw_dist) : 0.0f;
    const float px_scale = viewport_proj_scale_y * (float(resolution.y) * 0.5f);
    constexpr float min_screen_radius_px = 0.5f;

    const bool is_scatter = state.instance_mats.size() > 1;
    std::vector<std::pair<float, float4x4>> sorted_instances;
    sorted_instances.reserve(state.instance_mats.size());

    for (size_t inst_idx = 0; inst_idx < state.instance_mats.size(); ++inst_idx) {
      float4x4 inst_mat = state.instance_mats[inst_idx];
      if (tessera_have_bs) {
        const float3 inst_pos = (inst_mat * float4(tessera_bs_center, 1.0f)).xyz();
        const float dist_sq = math::distance_squared(tessera_cull_view.camPos, inst_pos);

        if (max_d_sq > 0.0f && dist_sq > max_d_sq) {
          continue;
        }

        if (px_scale > 0.0f && tessera_bs_radius > 0.0f) {
          const float inst_scale = std::max({math::length(inst_mat[0].xyz()),
                                             math::length(inst_mat[1].xyz()),
                                             math::length(inst_mat[2].xyz())});
          const float world_radius = tessera_bs_radius * inst_scale;
          const float d = std::sqrt(std::max(0.001f, dist_sq));
          const float screen_r_px = (world_radius * px_scale) / d;
          if (screen_r_px < min_screen_radius_px) {
            continue;
          }
        }

        if (!tessera::instance_in_frustum(tessera_cull_view,
                                          inst_mat,
                                          tessera_bs_center,
                                          tessera_bs_radius,
                                          state.culling_margin))
        {
          continue;
        }

        (void)is_scatter;
        sorted_instances.emplace_back(dist_sq, inst_mat);
      }
      else {
        sorted_instances.emplace_back(0.0f, inst_mat);
      }
    }

    std::sort(sorted_instances.begin(), sorted_instances.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (const auto &s_item : sorted_instances) {
      state.frame_visible_instances.push_back(s_item.second);
    }

    if (tess_eevee_dbg_enabled()) {
      tess_eevee_dbg_state("prepass_B", state);
    }
    /* Rasterize depth for visible scattered instances */
    const auto &draw_mats = state.frame_visible_instances;
    if (!draw_mats.empty()) {
      const float4x4 &ref_mat = draw_mats[0];
      const float4x4 ref_obj_persmat = viewport_persmat * ref_mat;
      const float4x4 ref_lod_obj_persmat = has_cam_lod ? (cam_persmat * ref_mat) : ref_obj_persmat;
      const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;
      const float error_scale = std::max({math::length(ref_mat[0].xyz()),
                                          math::length(ref_mat[1].xyz()),
                                          math::length(ref_mat[2].xyz())});

      tessera_expand_traversal_margin(*state.pass,
                                      state.data,
                                      ref_mat,
                                      draw_mats,
                                      tessera_cull_view,
                                      state.culling_margin);
      state.pass->update(state.data,
                         ref_obj_persmat,
                         viewport_proj_scale_y,
                         float(resolution.y),
                         has_cam_lod ? &ref_lod_obj_persmat : nullptr,
                         has_cam_lod ? &lod_proj_scale_y : nullptr,
                         uint32_t(draw_mats.size()),
                         error_scale);
      state.traversed_this_frame = true;

      state.pass->draw_depth_instances(draw_mats.data(),
                                       draw_mats.size(),
                                       resolution.x,
                                       resolution.y,
                                       viewport_persmat);
    }
  }
}

void TesseraModule::draw(const draw::View &view,
                         const int2 &resolution,
                         gpu::FrameBuffer *target_fb)
{
  if (objects_.is_empty()) {
    return;
  }
  std::lock_guard lock(this->mutex_);
  const float4x4 viewport_persmat = view.persmat();
  const float viewport_proj_scale_y = std::abs(view.winmat()[1][1]);
  /* World-space frustum planes for per-instance camera culling. */
  const tessera::TesseraViewInfo tessera_cull_view = tessera::getViewInfo(
      viewport_persmat, float(resolution.y), viewport_proj_scale_y);

  if (target_fb != nullptr) {
    GPU_framebuffer_bind(target_fb);
  }

  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty() || state.is_instancer_base) {
      continue;
    }
    /* Never dereference the pointer stored at sync time: the depsgraph may
     * have freed the evaluated object in between (procedural instancers and
     * scatter systems recreate objects every frame). Re-resolve by name. */
    Object *ob = (inst_.draw_ctx != nullptr) ?
                     tessera_object_revalidate(inst_.draw_ctx, state.ob_name) :
                     nullptr;
    if (ob == nullptr && !state.matched_archetype_name.empty() && inst_.draw_ctx != nullptr) {
      ob = tessera_object_revalidate(inst_.draw_ctx, state.matched_archetype_name);
    }
    if (ob == nullptr) {
      if (inst_.draw_ctx != nullptr) {
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
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    const int view_source = (state.ob != nullptr) ?
        tessera_view_source_get(state.ob) : state.view_source;
    state.view_source = view_source;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    const bool is_cam_view = (state.view_source != 0) ||
                             (inst_.rv3d != nullptr && inst_.rv3d->persp == RV3D_CAMOB);
    bool has_cam_lod = false;
    if (is_cam_view) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y))
      {
        has_cam_lod = true;
      }
    }
    const float4x4 lod_object_persmat = has_cam_lod ? (cam_persmat * state.object_mat) :
                                                      object_persmat;
    const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;

    if (state.pass == nullptr) {
      state.pass = std::make_unique<tessera::TesseraDrawPass>(
          state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
      if (!state.pass->init()) {
        state.pass.reset();
        continue;
      }
      /* EEVEE uses Reverse-Z depth testing and bucketed multi-draw indirect
       * args (like Workbench), so the counter buffer must be populated with
       * per-bucket DrawArraysIndirectCommand descriptors. */
      state.pass->setUseElementsIndirect(true);
    }
    /* EEVEE uses Reverse-Z depth testing. */
    state.pass->setReverseZ(true);
    const float culling_margin = (state.ob != nullptr) ?
        tessera_culling_margin_get(state.ob) : state.culling_margin;
    state.culling_margin = culling_margin;
    state.pass->setCullingMargin(culling_margin);
    const float pixel_error = (state.ob != nullptr) ?
        tessera_pixel_error_get(state.ob) : state.pixel_error;
    state.pixel_error = pixel_error;
    bool is_navigating_draw = (inst_.rv3d != nullptr) && (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || inst_.rv3d->viewlock != 0);
    if (inst_.scene != nullptr) {
      if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
        if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
          if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
            is_navigating_draw = true;
          }
        }
      }
    }
    if (!is_navigating_draw && inst_.rv3d == nullptr) {
      tessera_stat_culling_margin_write(state.ob, culling_margin);
    }
    float effective_pixel_error = pixel_error;
    if (is_navigating_draw) {
      effective_pixel_error *= 2.5f;
    }
    state.pass->setPixelError(effective_pixel_error);
    state.pass->setNavigating(is_navigating_draw);
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
    const bool show_stats = state.show_stats;
    const bool log_console = state.log_console;
    const bool show_fps_panel = state.show_fps_panel;
    const bool show_fps_hud = state.show_fps_hud;
    state.pass->setTelemetryEnabled(show_stats || log_console || show_fps_panel || show_fps_hud);
    if (!state.traversed_this_frame) {
      tessera_expand_traversal_margin(*state.pass,
                                      state.data,
                                      state.object_mat,
                                      state.instance_mats,
                                      tessera_cull_view,
                                      state.culling_margin);
      state.pass->update(state.data,
                         object_persmat,
                         viewport_proj_scale_y,
                         float(resolution.y),
                         has_cam_lod ? &lod_object_persmat : nullptr,
                         has_cam_lod ? &lod_proj_scale_y : nullptr,
                         uint32_t(state.instance_mats.size()),
                         1.0f);
      state.traversed_this_frame = true;
    }

    update_telemetry(state, view.location());

    state.pass->setTextureScale(state.current_tex_scale);
    state.pass->set_material_textures(state.materials);
    if (state.pass->drawCount() != 0) {
      state.pass->set_two_sided(state.two_sided);
      if (!state.instance_mats.empty()) {
        const auto &draw_mats = !state.frame_visible_instances.empty() ?
                                    state.frame_visible_instances :
                                    state.instance_mats;
        if (!draw_mats.empty()) {
          const float4 base_col = state.base_color_valid ? state.base_color :
                                                           float4(0.8f, 0.8f, 0.8f, 1.0f);
          state.pass->draw_forward_instances(draw_mats.data(),
                                             draw_mats.size(),
                                             resolution.x,
                                             resolution.y,
                                             1,
                                             base_col,
                                             viewport_persmat);
        }
      }
      else {
        state.pass->draw(resolution.x, resolution.y);
      }
    }
  }
}

bool TesseraModule::has_objects() const
{
  std::lock_guard lock(const_cast<std::mutex &>(this->mutex_));
  for (const auto item : objects_.items()) {
    if (item.value->used && !item.value->data.clusters.empty()) {
      return true;
    }
  }
  return false;
}

bool TesseraModule::has_cluster_visualization() const
{
  std::lock_guard lock(const_cast<std::mutex &>(this->mutex_));
  for (const auto item : objects_.items()) {
    if (item.value->used && !item.value->data.clusters.empty() && item.value->visualize_clusters > 0) {
      return true;
    }
  }
  return false;
}

void TesseraModule::draw_gbuffer(const draw::View &view,
                                 const int2 &resolution,
                                 gpu::FrameBuffer *target_fb)
{
  if (objects_.is_empty()) {
    return;
  }
  std::lock_guard lock(this->mutex_);
  const float4x4 viewport_persmat = view.persmat();
  const float viewport_proj_scale_y = std::abs(view.winmat()[1][1]);
  const tessera::TesseraViewInfo tessera_cull_view = tessera::getViewInfo(
      viewport_persmat, float(resolution.y), viewport_proj_scale_y);

  if (target_fb != nullptr) {
    GPU_framebuffer_bind(target_fb);
  }

  /* Phase A: Terrain and single meshes first */
  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty() || state.is_instancer_base || !state.instance_mats.empty()) {
      continue;
    }
    Object *ob = (inst_.draw_ctx != nullptr) ?
                     tessera_object_revalidate(inst_.draw_ctx, state.ob_name) :
                     nullptr;
    if (ob == nullptr && !state.matched_archetype_name.empty() && inst_.draw_ctx != nullptr) {
      ob = tessera_object_revalidate(inst_.draw_ctx, state.matched_archetype_name);
    }
    if (ob == nullptr) {
      if (inst_.draw_ctx != nullptr) {
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
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    const int view_source = (state.ob != nullptr) ?
        tessera_view_source_get(state.ob) : state.view_source;
    state.view_source = view_source;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    const bool is_cam_view = (state.view_source != 0) ||
                             (inst_.rv3d != nullptr && inst_.rv3d->persp == RV3D_CAMOB);
    bool has_cam_lod = false;
    if (is_cam_view) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y))
      {
        has_cam_lod = true;
      }
    }
    const float4x4 lod_object_persmat = has_cam_lod ? (cam_persmat * state.object_mat) :
                                                      object_persmat;
    const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;

    if (state.pass == nullptr) {
      state.pass = std::make_unique<tessera::TesseraDrawPass>(
          state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
      if (!state.pass->init()) {
        state.pass.reset();
        continue;
      }
    }
    state.pass->setReverseZ(true);
    const float culling_margin = (state.ob != nullptr) ?
        tessera_culling_margin_get(state.ob) : state.culling_margin;
    state.culling_margin = culling_margin;
    state.pass->setCullingMargin(culling_margin);
    const float pixel_error = (state.ob != nullptr) ?
        tessera_pixel_error_get(state.ob) : state.pixel_error;
    state.pixel_error = pixel_error;
    bool is_navigating_gbuf = (inst_.rv3d != nullptr) && (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || inst_.rv3d->viewlock != 0);
    if (inst_.scene != nullptr) {
      if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
        if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
          if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
            is_navigating_gbuf = true;
          }
        }
      }
    }
    if (!is_navigating_gbuf && inst_.rv3d == nullptr) {
      tessera_stat_culling_margin_write(state.ob, culling_margin);
    }
    float effective_pixel_error = pixel_error;
    if (is_navigating_gbuf) {
      effective_pixel_error *= 2.5f;
    }
    state.pass->setPixelError(effective_pixel_error);
    state.pass->setNavigating(is_navigating_gbuf);
    state.pass->setTelemetryEnabled(state.show_stats || state.log_console || state.show_fps_panel || state.show_fps_hud);

    if (!state.traversed_this_frame) {
      state.pass->update(state.data,
                         object_persmat,
                         viewport_proj_scale_y,
                         float(resolution.y),
                         has_cam_lod ? &lod_object_persmat : nullptr,
                         has_cam_lod ? &lod_proj_scale_y : nullptr);
      state.traversed_this_frame = true;
    }
    update_telemetry(state, view.location());
    state.pass->setTextureScale(state.current_tex_scale);
    state.pass->set_material_textures(state.materials);
    if (state.pass->drawCount() != 0) {
      state.pass->set_two_sided(state.two_sided);
      state.pass->draw_gbuffer(resolution.x, resolution.y, state.object_mat);
    }
  }

  /* Phase B: Scattered instance fields second */
  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty() || state.is_instancer_base || state.instance_mats.empty()) {
      continue;
    }
    Object *ob = (inst_.draw_ctx != nullptr) ?
                     tessera_object_revalidate(inst_.draw_ctx, state.ob_name) :
                     nullptr;
    if (ob == nullptr && !state.matched_archetype_name.empty() && inst_.draw_ctx != nullptr) {
      ob = tessera_object_revalidate(inst_.draw_ctx, state.matched_archetype_name);
    }
    if (ob == nullptr) {
      if (inst_.draw_ctx != nullptr) {
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
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    const int view_source = (state.ob != nullptr) ?
        tessera_view_source_get(state.ob) : state.view_source;
    state.view_source = view_source;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    const bool is_cam_view = (state.view_source != 0) ||
                             (inst_.rv3d != nullptr && inst_.rv3d->persp == RV3D_CAMOB);
    bool has_cam_lod = false;
    if (is_cam_view) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, state.ob, cam_persmat, cam_proj_scale_y))
      {
        has_cam_lod = true;
      }
    }

    if (state.pass == nullptr) {
      state.pass = std::make_unique<tessera::TesseraDrawPass>(
          state.data, object_persmat, viewport_proj_scale_y, float(resolution.y));
      if (!state.pass->init()) {
        state.pass.reset();
        continue;
      }
    }
    state.pass->setReverseZ(true);
    const float culling_margin = (state.ob != nullptr) ?
        tessera_culling_margin_get(state.ob) : state.culling_margin;
    state.culling_margin = culling_margin;
    state.pass->setCullingMargin(culling_margin);
    const float pixel_error = (state.ob != nullptr) ?
        tessera_pixel_error_get(state.ob) : state.pixel_error;
    state.pixel_error = pixel_error;
    bool is_navigating_gbuf = (inst_.rv3d != nullptr) && (((inst_.rv3d->rflag & RV3D_NAVIGATING) != 0) || inst_.rv3d->viewlock != 0);
    if (inst_.scene != nullptr) {
      if (IDProperty *sgroup = IDP_GetProperties(&inst_.scene->id)) {
        if (IDProperty *prop = IDP_GetPropertyFromGroup(sgroup, "tessera_navigating")) {
          if (prop->type == IDP_INT && IDP_int_get(prop) != 0) {
            is_navigating_gbuf = true;
          }
        }
      }
    }
    if (!is_navigating_gbuf && inst_.rv3d == nullptr) {
      tessera_stat_culling_margin_write(state.ob, culling_margin);
    }
    float effective_pixel_error = pixel_error;
    if (is_navigating_gbuf) {
      effective_pixel_error *= 2.5f;
    }
    state.pass->setPixelError(effective_pixel_error);
    state.pass->setNavigating(is_navigating_gbuf);
    state.pass->setTelemetryEnabled(state.show_stats || state.log_console || state.show_fps_panel || state.show_fps_hud);

    /* If prepass was bypassed, run frustum & distance culling for scatter instances */
    if (state.frame_visible_instances.empty() && !state.traversed_this_frame) {
      float3 tessera_bs_center(0.0f);
      float tessera_bs_radius = 0.0f;
      const bool tessera_have_bs = tessera::mesh_bounds_sphere(
          state.data, tessera_bs_center, tessera_bs_radius);
      const float max_draw_dist = eevee_max_draw_dist_get(state.ob_name, state.matched_archetype_name);
      const float max_d_sq = max_draw_dist > 0.0f ? (max_draw_dist * max_draw_dist) : 0.0f;
      const float px_scale = viewport_proj_scale_y * (float(resolution.y) * 0.5f);
      constexpr float min_screen_radius_px = 0.5f;

      for (const float4x4 &inst_mat : state.instance_mats) {
        if (tessera_have_bs) {
          const float3 inst_pos = (inst_mat * float4(tessera_bs_center, 1.0f)).xyz();
          const float dist_sq = math::distance_squared(tessera_cull_view.camPos, inst_pos);
          if (max_d_sq > 0.0f && dist_sq > max_d_sq) continue;
          if (px_scale > 0.0f && tessera_bs_radius > 0.0f) {
            const float inst_scale = std::max({math::length(inst_mat[0].xyz()),
                                               math::length(inst_mat[1].xyz()),
                                               math::length(inst_mat[2].xyz())});
            const float world_radius = tessera_bs_radius * inst_scale;
            const float d = std::sqrt(std::max(0.001f, dist_sq));
            if ((world_radius * px_scale) / d < min_screen_radius_px) continue;
          }
          if (!tessera::instance_in_frustum(tessera_cull_view, inst_mat, tessera_bs_center, tessera_bs_radius, state.culling_margin)) continue;
        }
        state.frame_visible_instances.push_back(inst_mat);
      }
    }

    if (tess_eevee_dbg_enabled()) {
      tess_eevee_dbg_state("gbuffer_B", state);
      if (state.pass != nullptr) {
        tess_eevee_log("[tess-eevee] gbuffer_B pass: drawCount=%u\n", state.pass->drawCount());
      }
      if (!state.instance_mats.empty()) {
        const float4x4 &m0 = state.instance_mats[0];
        tess_eevee_log("[tess-eevee] gbuffer_B m0: t=%.3f,%.3f,%.3f s=%.3f,%.3f,%.3f\n",
                       m0[3][0], m0[3][1], m0[3][2],
                       math::length(m0[0].xyz()), math::length(m0[1].xyz()), math::length(m0[2].xyz()));
      }
    }
    const auto &draw_mats = state.frame_visible_instances;
    if (!draw_mats.empty()) {
      if (!state.traversed_this_frame) {
        const float4x4 &ref_mat = draw_mats[0];
        const float4x4 ref_obj_persmat = viewport_persmat * ref_mat;
        const float4x4 ref_lod_obj_persmat = has_cam_lod ? (cam_persmat * ref_mat) : ref_obj_persmat;
        const float lod_proj_scale_y = has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y;
        const float error_scale = std::max({math::length(ref_mat[0].xyz()),
                                            math::length(ref_mat[1].xyz()),
                                            math::length(ref_mat[2].xyz())});

        tessera_expand_traversal_margin(*state.pass,
                                        state.data,
                                        ref_mat,
                                        draw_mats,
                                        tessera_cull_view,
                                        state.culling_margin);
        state.pass->update(state.data,
                           ref_obj_persmat,
                           viewport_proj_scale_y,
                           float(resolution.y),
                           has_cam_lod ? &ref_lod_obj_persmat : nullptr,
                           has_cam_lod ? &lod_proj_scale_y : nullptr,
                           uint32_t(draw_mats.size()),
                           error_scale);
        state.traversed_this_frame = true;
      }

      update_telemetry(state, view.location());
      state.pass->setTextureScale(state.current_tex_scale);
      state.pass->set_material_textures(state.materials);
      state.pass->set_two_sided(state.two_sided);

      bool has_diffuse_tex = false;
      for (const auto &m : state.materials) {
        if (m.has_diffuse && m.diffuse != nullptr) {
          has_diffuse_tex = true;
          break;
        }
      }
      state.pass->set_alpha_mask(has_diffuse_tex);

      state.pass->draw_gbuffer_instances(draw_mats.data(),
                                         draw_mats.size(),
                                         resolution.x,
                                         resolution.y,
                                         viewport_persmat);
    }
  }
}

void TesseraModule::draw_shadow(const draw::View &shadow_view,
                                gpu::FrameBuffer *shadow_fb)
{
  (void)shadow_view;
  (void)shadow_fb;
}

}  // namespace blender::eevee
