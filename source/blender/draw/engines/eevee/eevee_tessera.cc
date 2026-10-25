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

#include "DNA_camera_types.h"
#include "DNA_ID.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

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
#include <map>
#include <string>

namespace blender::eevee {

TesseraModule::TesseraModule(Instance &inst) : inst_(inst) {}

TesseraModule::~TesseraModule() = default;

void TesseraModule::begin_sync()
{
  std::lock_guard lock(this->mutex_);
  for (auto item : objects_.items()) {
    item.value->used = false;
  }
}

static IDProperty *tessera_property_find(Object *ob, const char *prop_name)
{
  if (ob == nullptr) {
    return nullptr;
  }
  IDProperty *group = IDP_GetProperties(&ob->id);
  if (group != nullptr) {
    if (IDProperty *prop = IDP_GetPropertyFromGroup(group, prop_name)) {
      return prop;
    }
  }
  const ID *orig_id = DEG_get_original_id(&ob->id);
  if (orig_id != nullptr && orig_id != &ob->id) {
    IDProperty *orig_group = IDP_GetProperties(const_cast<ID *>(orig_id));
    if (orig_group != nullptr) {
      if (IDProperty *prop = IDP_GetPropertyFromGroup(orig_group, prop_name)) {
        return prop;
      }
    }
  }
  return nullptr;
}

static bool tessera_path_get(Object *ob, std::string &r_path)
{
  const IDProperty *prop = tessera_property_find(ob, "tessera_nc2");
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
  const IDProperty *prop = tessera_property_find(ob, "tessera_visualize_clusters");
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

static bool tessera_enabled_get(Object *ob)
{
  const IDProperty *prop = tessera_property_find(ob, "tessera_enabled");
  if (prop == nullptr) {
    return true;
  }
  if (prop->type == IDP_INT) {
    return IDP_int_get(prop) != 0;
  }
  return prop->data.val != 0;
}

static float tessera_pixel_error_get(Object *ob)
{
  const IDProperty *prop = tessera_property_find(ob, "tessera_pixel_error");
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

static int tessera_view_source_get(Object *ob)
{
  const IDProperty *prop = tessera_property_find(ob, "tessera_view_source");
  if (prop == nullptr) {
    return 0;
  }
  if (prop->type == IDP_INT) {
    return IDP_int_get(prop);
  }
  return prop->data.val;
}

static bool tessera_show_stats_get(Object *ob)
{
  const IDProperty *prop = tessera_property_find(ob, "tessera_show_stats");
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
  const IDProperty *prop = tessera_property_find(ob, "tessera_log_console");
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
  const IDProperty *prop = tessera_property_find(ob, "tessera_show_fps_panel");
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
  const IDProperty *prop = tessera_property_find(ob, "tessera_show_fps_hud");
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
  if (idgroup == nullptr) {
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
  if (idgroup == nullptr) {
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

static void tessera_write_telemetry_to_object(Object *ob, const tessera::TesseraTelemetry &telem)
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
                                  float4x4 &r_persmat,
                                  float &r_proj_scale_y)
{
  if (scene == nullptr || scene->camera == nullptr || draw_ctx->depsgraph == nullptr) {
    return false;
  }
  const Object *cam_ob = DEG_get_evaluated(draw_ctx->depsgraph, scene->camera);
  if (cam_ob == nullptr || cam_ob->type != OB_CAMERA || cam_ob->data == nullptr) {
    return false;
  }
  CameraParams params;
  BKE_camera_params_init(&params);
  BKE_camera_params_from_object(&params, cam_ob);
  BKE_camera_params_compute_viewplane(
      &params, resolution.x, resolution.y, scene->r.xasp, scene->r.yasp);
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

    if (!ma->use_nodes || ma->nodetree == nullptr) {
      continue;
    }

    for (bNode *node = static_cast<bNode *>(ma->nodetree->nodes.first); node != nullptr;
         node = node->next)
    {
      if (node->type_legacy != SH_NODE_TEX_IMAGE || node->id == nullptr) {
        continue;
      }
      blender::Image *ima = reinterpret_cast<blender::Image *>(node->id);
      NodeTexImage *storage = static_cast<NodeTexImage *>(node->storage);
      ImageUser *iuser = storage ? &storage->iuser : nullptr;
      gpu::Texture *tex = BKE_image_get_gpu_texture(ima, iuser);
      if (tex == nullptr) {
        ImageGPUTextures gputex = BKE_image_get_gpu_material_texture(ima, iuser, false);
        tex = gputex.texture ? *gputex.texture : nullptr;
      }
      if (tex == nullptr) {
        continue;
      }

      std::string node_name = node->name;
      std::string ima_name = ima->id.name + 2;
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
      }
      else if (!mat.has_diffuse && !is_normal && !is_rough) {
        mat.diffuse = tex;
        mat.has_diffuse = true;
      }
    }
  }
  return result;
}

bool TesseraModule::record(Object *ob)
{
  if (!tessera_enabled_get(ob)) {
    return false;
  }
  std::string idprop_path;
  if (!tessera_path_get(ob, idprop_path)) {
    return false;
  }

  char abs_path[FILE_MAX];
  BLI_strncpy(abs_path, idprop_path.c_str(), sizeof(abs_path));
  BLI_path_abs(abs_path, BKE_main_blendfile_path_from_global());

  BLI_stat_t file_info;
  const bool stat_ok = (BLI_stat(abs_path, &file_info) == 0);
  const int64_t file_mtime = stat_ok ? int64_t(file_info.st_mtime) : int64_t(-1);
  const int64_t file_size = stat_ok ? int64_t(file_info.st_size) : int64_t(-1);

  std::lock_guard lock(this->mutex_);
  std::unique_ptr<ObjectState> &entry = objects_.lookup_or_add_default(
      std::string(ob->id.name + 2));
  if (entry == nullptr) {
    entry = std::make_unique<ObjectState>();
  }
  ObjectState &state = *entry;

  state.used = true;
  state.ob = ob;
  state.show_stats = tessera_show_stats_get(ob);
  state.log_console = tessera_log_console_get(ob);
  state.show_fps_panel = tessera_show_fps_panel_get(ob);
  state.show_fps_hud = tessera_show_fps_hud_get(ob);
  state.visualize_clusters = tessera_visualize_get(ob);
  state.pixel_error = tessera_pixel_error_get(ob);
  state.view_source = tessera_view_source_get(ob);
  state.object_mat = ob->object_to_world();
  state.materials = tessera_extract_materials(ob);

  const bool stale = (state.nc2_path != abs_path) ||
                     (state.file_mtime != file_mtime) ||
                     (state.file_size != file_size);
  if (state.pass == nullptr || stale) {
    if (stale || state.data.clusters.empty()) {
      size_t data_size = 0;
      if (void *file_data = BLI_file_read_binary_as_mem(abs_path, 0, &data_size)) {
        tessera::TesseraData loaded;
        if (tessera::deserializeFromMemory(
                loaded, static_cast<const uint8_t *>(file_data), data_size) &&
            !loaded.clusters.empty())
        {
          state.data = std::move(loaded);
        }
        MEM_delete_void(file_data);
      }
      state.file_mtime = file_mtime;
      state.file_size = file_size;
      state.nc2_path = abs_path;
      state.pass.reset();
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

  if (!show_stats && !log_console && !show_fps_panel && !show_fps_hud) {
    state.last_frame_time = {};
    state.smoothed_dt = 0.0f;
    return;
  }

  /* Compute FPS / frame time metrics if requested. */
  if (show_fps_panel || show_fps_hud) {
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
    if (state.ob != nullptr) {
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

  /* Extract telemetry counters and update object properties. */
  if (show_stats || log_console || show_fps_hud) {
    using namespace std::chrono;
    auto now = steady_clock::now();
    auto elapsed_ms = (state.last_stat_time.time_since_epoch().count() == 0) ?
                          60 :
                          duration_cast<milliseconds>(now - state.last_stat_time).count();
    if (elapsed_ms >= 60) {
      state.last_stat_time = now;
      const float3 bounds_min(state.data.header.boundsMin[0],
                              state.data.header.boundsMin[1],
                              state.data.header.boundsMin[2]);
      const float3 bounds_max(state.data.header.boundsMax[0],
                              state.data.header.boundsMax[1],
                              state.data.header.boundsMax[2]);
      const float3 local_center = 0.5f * (bounds_min + bounds_max);
      const float3 world_center = (state.object_mat * float4(local_center, 1.0f)).xyz();
      const float cam_distance = math::distance(cam_pos, world_center);

      tessera::TesseraTelemetry telem = state.pass ? state.pass->getTelemetry() :
                                                     tessera::TesseraTelemetry{};
      telem.cam_distance = cam_distance;
      telem.pixel_error = state.pass ? state.pass->getPixelError() : state.pixel_error;

      if ((show_stats || show_fps_hud) && state.ob != nullptr) {
        tessera_write_telemetry_to_object(state.ob, telem);
        WM_main_add_notifier(NC_OBJECT | ND_DRAW, state.ob);
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

  if (depth_fb != nullptr) {
    GPU_framebuffer_bind(depth_fb);
  }

  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty()) {
      continue;
    }
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    bool has_cam_lod = false;
    if (state.view_source != 0) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, cam_persmat, cam_proj_scale_y))
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
    /* EEVEE uses Reverse-Z depth testing. */
    state.pass->setReverseZ(true);
    state.pass->setPixelError(state.pixel_error);
    const bool show_stats = state.show_stats;
    const bool log_console = state.log_console;
    const bool show_fps_panel = state.show_fps_panel;
    const bool show_fps_hud = state.show_fps_hud;
    state.pass->setTelemetryEnabled(show_stats || log_console || show_fps_panel || show_fps_hud);
    state.pass->update(state.data,
                       object_persmat,
                       viewport_proj_scale_y,
                       float(resolution.y),
                       has_cam_lod ? &lod_object_persmat : nullptr,
                       has_cam_lod ? &lod_proj_scale_y : nullptr);

    if (state.pass->drawCount() != 0) {
      state.pass->draw_prepass(resolution.x, resolution.y);
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

  if (target_fb != nullptr) {
    GPU_framebuffer_bind(target_fb);
  }

  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty()) {
      continue;
    }
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    bool has_cam_lod = false;
    if (state.view_source != 0) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, cam_persmat, cam_proj_scale_y))
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
    /* EEVEE uses Reverse-Z depth testing. */
    state.pass->setReverseZ(true);
    state.pass->setPixelError(state.pixel_error);
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
    state.pass->update(state.data,
                       object_persmat,
                       viewport_proj_scale_y,
                       float(resolution.y),
                       has_cam_lod ? &lod_object_persmat : nullptr,
                       has_cam_lod ? &lod_proj_scale_y : nullptr);

    update_telemetry(state, view.location());

    state.pass->set_material_textures(state.materials);
    if (state.pass->drawCount() != 0) {
      state.pass->draw(resolution.x, resolution.y);
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
                                 gpu::FrameBuffer *gbuffer_fb)
{
  if (objects_.is_empty()) {
    return;
  }
  std::lock_guard lock(this->mutex_);
  const float4x4 viewport_persmat = view.persmat();
  const float viewport_proj_scale_y = std::abs(view.winmat()[1][1]);

  if (gbuffer_fb != nullptr) {
    GPU_framebuffer_bind(gbuffer_fb);
  }

  for (auto item : objects_.items()) {
    ObjectState &state = *item.value;
    if (!state.used || state.data.clusters.empty()) {
      continue;
    }
    const float4x4 object_persmat = viewport_persmat * state.object_mat;

    float4x4 cam_persmat;
    float cam_proj_scale_y = viewport_proj_scale_y;
    bool has_cam_lod = false;
    if (state.view_source != 0) {
      if (tessera_camera_persmat(
              inst_.scene, inst_.draw_ctx, resolution, cam_persmat, cam_proj_scale_y))
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
    /* EEVEE uses Reverse-Z depth testing. */
    state.pass->setReverseZ(true);
    state.pass->setPixelError(state.pixel_error);
    const bool show_stats = state.show_stats;
    const bool log_console = state.log_console;
    const bool show_fps_panel = state.show_fps_panel;
    const bool show_fps_hud = state.show_fps_hud;
    state.pass->setTelemetryEnabled(show_stats || log_console || show_fps_panel || show_fps_hud);
    state.pass->update(state.data,
                       object_persmat,
                       viewport_proj_scale_y,
                       float(resolution.y),
                       has_cam_lod ? &lod_object_persmat : nullptr,
                       has_cam_lod ? &lod_proj_scale_y : nullptr);

    update_telemetry(state, view.location());

    state.pass->set_material_textures(state.materials);
    if (state.pass->drawCount() != 0) {
      state.pass->draw_gbuffer(resolution.x, resolution.y, state.object_mat);
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
