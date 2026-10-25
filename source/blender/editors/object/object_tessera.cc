/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 * \ingroup edobj
 *
 * Tessera build operator: the C<->Python bridge.
 *
 * `bpy.ops.tessera.build` (scripts/modules/tessera/build.py) delegates here so
 * the real C++ cluster DAG builder (QEM simplification) runs instead of the
 * Python placeholder.
 */

#include <cstdio>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "MEM_guardedalloc.h"

#include "BLI_math_vector_types.hh"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_listbase.h"
#include "BLI_path_utils.hh"
#include "BLI_fileops.hh"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "BKE_context.hh"
#include "BKE_global.hh"
#include "BKE_idprop.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_report.hh"
#include "BKE_workspace.hh"

#include "DNA_screen_types.h"
#include "DNA_workspace_types.h"
#include "ED_object.hh"
#include "ED_screen.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "WM_api.hh"
#include "WM_types.hh"
#include "wm_event_types.hh"

#include "tessera.h"
#include "tessera_draw.hh"

#include "GPU_context.hh"
#include "GPU_framebuffer.hh"
#include "GPU_init_exit.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"

#include "object_intern.hh"

#include "DEG_depsgraph.hh"

namespace blender::ed::object {

/* -------------------------------------------------------------------- */
/** \name Tessera Build Operator
 * \{ */

struct TesseraBuildJob {
  Main *bmain = nullptr;
  std::string ob_name;
  Object *ob = nullptr;
  Scene *scene = nullptr;
  tessera::TesseraRawMesh raw;
  char filepath[FILE_MAX] = "";
  char stored_path[FILE_MAX] = "";
  tessera::TesseraData data;
  std::vector<uint8_t> blob;
  bool success = false;
  bool is_nonblocking_job = false;
  ReportList *reports = nullptr;
  wmJob *wm_job = nullptr;
  wmJobWorkerStatus *worker_status = nullptr;

  std::atomic<float> progress{0.0f};
  std::mutex status_mutex;
  std::string status_name;
};

static void tessera_build_start_job(void *customdata, wmJobWorkerStatus *worker_status)
{
  TesseraBuildJob *job = static_cast<TesseraBuildJob *>(customdata);
  job->worker_status = worker_status;

  auto progress_cb = [job, worker_status](float progress, const char *step_name) {
    job->progress.store(progress, std::memory_order_relaxed);
    if (step_name && step_name[0] != '\0') {
      std::lock_guard lock(job->status_mutex);
      job->status_name = step_name;
    }
    if (worker_status) {
      worker_status->progress = progress;
      worker_status->do_update = true;
    }
    if (job->wm_job) {
      WM_jobs_progress(job->wm_job, progress);
      if (step_name && step_name[0] != '\0') {
        WM_jobs_status_name(job->wm_job, step_name);
      }
    }
  };

  if (!tessera::buildFromRawMesh(job->raw, job->data, progress_cb)) {
    if (worker_status && worker_status->reports) {
      BKE_report(worker_status->reports, RPT_ERROR, "Tessera: failed to build cluster DAG from mesh");
    }
    return;
  }

  job->progress.store(0.95f, std::memory_order_relaxed);
  {
    std::lock_guard lock(job->status_mutex);
    job->status_name = "Saving Tessera .nc2 file...";
  }
  if (worker_status) {
    worker_status->progress = 0.95f;
    worker_status->do_update = true;
  }
  if (job->wm_job) {
    WM_jobs_progress(job->wm_job, 0.95f);
    WM_jobs_status_name(job->wm_job, "Saving Tessera .nc2 file...");
  }

  if (!tessera::serializeToMemory(job->data, job->blob)) {
    if (worker_status && worker_status->reports) {
      BKE_report(worker_status->reports, RPT_ERROR, "Tessera: failed to serialize cluster data");
    }
    return;
  }

  FILE *file = BLI_fopen(job->filepath, "wb");
  if (file == nullptr) {
    if (worker_status && worker_status->reports) {
      BKE_reportf(
          worker_status->reports, RPT_ERROR, "Tessera: could not open '%s' for writing", job->filepath);
    }
    return;
  }
  const size_t written = fwrite(job->blob.data(), 1, job->blob.size(), file);
  fclose(file);
  if (written != job->blob.size()) {
    if (worker_status && worker_status->reports) {
      BKE_reportf(worker_status->reports, RPT_ERROR, "Tessera: short write to '%s'", job->filepath);
    }
    return;
  }

  job->success = true;
  job->progress.store(1.0f, std::memory_order_relaxed);
  {
    std::lock_guard lock(job->status_mutex);
    job->status_name = "Tessera clusters built!";
  }
  if (worker_status) {
    worker_status->progress = 1.0f;
    worker_status->do_update = true;
  }
  if (job->wm_job) {
    WM_jobs_progress(job->wm_job, 1.0f);
    WM_jobs_status_name(job->wm_job, "Tessera clusters built!");
  }
}

static void idp_string_replace(IDProperty *idgroup, const char *name, const char *value)
{
  if (idgroup == nullptr) {
    return;
  }
  if (IDProperty *prev = IDP_GetPropertyFromGroup(idgroup, name)) {
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

static Object *tessera_find_job_object(Main *bmain, const std::string &ob_name)
{
  if (bmain == nullptr || ob_name.empty()) {
    return nullptr;
  }

  /* Safe C++ range-based traversal over blender::Main::objects */
  for (Object &ob : bmain->objects) {
    /* Compare against both "OB<name>" and raw "<name>" */
    if (ob.id.name + 2 == ob_name || std::string(ob.id.name) == ob_name) {
      return &ob;
    }
  }

  return nullptr;
}

static void tessera_build_end_job(void *customdata)
{
  TesseraBuildJob *job = static_cast<TesseraBuildJob *>(customdata);
  if (job == nullptr) {
    return;
  }

  Main *bmain = job->bmain != nullptr ? job->bmain : G_MAIN;
  Object *ob = tessera_find_job_object(bmain, job->ob_name);
  if (ob != nullptr) {
    if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
      idp_int_replace(idgroup, "tessera_building", 0);
      idp_float_replace(idgroup, "tessera_build_progress", 1.0f);
      idp_string_replace(idgroup,
                         "tessera_build_status",
                         job->success ? "Tessera clusters built!" : "Build failed");
      if (job->success) {
        idp_string_replace(idgroup, "tessera_nc2", job->stored_path);
      }
    }
  }

  if (!job->success || ob == nullptr) {
    return;
  }

  /* Tag depsgraph and notify 3D Viewport to auto-refresh immediately. */
  DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY | ID_RECALC_SHADING);
  WM_main_add_notifier(NC_OBJECT | ND_DRAW, ob);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_VIEW3D, nullptr);

  if (job->reports) {
    BKE_reportf(job->reports,
                RPT_INFO,
                "Tessera: built %u clusters, %u levels (%zu bytes) -> %s",
                uint(job->data.clusters.size()),
                job->data.header.levelCount,
                job->blob.size(),
                job->filepath);
  }
}

static void tessera_build_free_job(void *customdata)
{
  TesseraBuildJob *job = static_cast<TesseraBuildJob *>(customdata);
  if (job != nullptr) {
    Main *bmain = job->bmain != nullptr ? job->bmain : G_MAIN;
    Object *ob = tessera_find_job_object(bmain, job->ob_name);
    if (ob != nullptr) {
      if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
        idp_int_replace(idgroup, "tessera_building", 0);
      }
      WM_main_add_notifier(NC_OBJECT | ND_DRAW, ob);
    }
    WM_main_add_notifier(NC_SPACE | ND_SPACE_VIEW3D, nullptr);
    delete job;
  }
}

static bool object_tessera_build_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr || ob->type != OB_MESH || ob->data == nullptr) {
    return false;
  }
  IDProperty *idgroup = IDP_GetProperties(&ob->id);
  if (idgroup != nullptr) {
    if (IDProperty *prop = IDP_GetPropertyFromGroup(idgroup, "tessera_building")) {
      if (IDP_int_get(prop) != 0) {
        return false;
      }
    }
  }
  return true;
}

static void tessera_tag_sidebar_redraw(bContext *C)
{
  wmWindowManager *wm = CTX_wm_manager(C);
  if (wm == nullptr) {
    return;
  }

  for (wmWindow *win = static_cast<wmWindow *>(wm->windows.first); win != nullptr;
       win = win->next)
  {
    WM_window_status_area_tag_redraw(win);
    bScreen *screen = BKE_workspace_active_screen_get(win->workspace_hook);
    if (screen == nullptr) {
      continue;
    }

    for (ScrArea *area = static_cast<ScrArea *>(screen->areabase.first); area != nullptr;
         area = area->next)
    {
      if (area->spacetype == SPACE_VIEW3D) {
        for (ARegion *region = static_cast<ARegion *>(area->regionbase.first); region != nullptr;
             region = region->next)
        {
          if (region->regiontype == RGN_TYPE_UI) {
            ED_region_tag_redraw(region);
          }
        }
      }
    }
  }
}

static void tessera_tag_redraw_all(bContext *C, Object *ob)
{
  tessera_tag_sidebar_redraw(C);
  if (ob != nullptr) {
    WM_main_add_notifier(NC_OBJECT | ND_DRAW, ob);
  }
  WM_main_add_notifier(NC_SPACE | ND_SPACE_VIEW3D, nullptr);
  WM_main_add_notifier(NC_WINDOW, nullptr);
}

static wmOperatorStatus object_tessera_build_modal(bContext *C,
                                                  wmOperator *op,
                                                  const wmEvent *event)
{
  TesseraBuildJob *job = static_cast<TesseraBuildJob *>(op->customdata);
  if (job == nullptr) {
    return OPERATOR_FINISHED;
  }

  Main *bmain = CTX_data_main(C);
  if (bmain == nullptr) {
    return OPERATOR_PASS_THROUGH;
  }

  Object *ob = tessera_find_job_object(bmain, job->ob_name);
  if (ob == nullptr) {
    /* Also check if the active object in context matches the job's target */
    Object *active_ob = CTX_data_active_object(C);
    if (active_ob != nullptr &&
        (active_ob->id.name + 2 == job->ob_name || std::string(active_ob->id.name) == job->ob_name))
    {
      ob = active_ob;
    }
  }

  if (ob == nullptr) {
    /* Object no longer exists in scene; cleanly abort */
    return OPERATOR_CANCELLED;
  }

  if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
    const float prog = job->progress.load(std::memory_order_relaxed);
    std::string status;
    {
      std::lock_guard lock(job->status_mutex);
      status = job->status_name;
    }
    idp_float_replace(idgroup, "tessera_build_progress", prog);
    if (!status.empty()) {
      idp_string_replace(idgroup, "tessera_build_status", status.c_str());
    }
    tessera_tag_redraw_all(C, ob);
  }

  Scene *scene = CTX_data_scene(C);
  wmWindowManager *wm = CTX_wm_manager(C);

  if (!WM_jobs_test(wm, scene, WM_JOB_TYPE_OBJECT_BAKE)) {
    if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
      idp_int_replace(idgroup, "tessera_building", 0);
    }
    tessera_tag_redraw_all(C, ob);
    return OPERATOR_FINISHED | OPERATOR_PASS_THROUGH;
  }
  if (event->type == EVT_ESCKEY && event->val == KM_PRESS) {
    WM_jobs_kill_type(wm, scene, WM_JOB_TYPE_OBJECT_BAKE);
    if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
      idp_int_replace(idgroup, "tessera_building", 0);
      idp_string_replace(idgroup, "tessera_build_status", "Cancelled");
    }
    tessera_tag_redraw_all(C, ob);
    return OPERATOR_CANCELLED;
  }
  return OPERATOR_PASS_THROUGH;
}

static wmOperatorStatus object_tessera_build_custom(bContext *C, wmOperator *op, bool is_invoke)
{
  Object *ob = CTX_data_active_object(C);
  Mesh *mesh = id_cast<Mesh *>(ob->data);
  Scene *scene = CTX_data_scene(C);
  Main *bmain = CTX_data_main(C);

  auto *job = new TesseraBuildJob();
  job->bmain = bmain;
  job->ob_name = ob->id.name + 2;
  job->ob = ob;
  job->scene = scene;
  job->reports = op->reports;

  /* Decoupled geometry extraction runs quickly on main thread. */
  if (!tessera::extractMeshGeometry(*mesh, job->raw)) {
    BKE_report(op->reports, RPT_ERROR, "Tessera: failed to extract mesh geometry");
    delete job;
    return OPERATOR_CANCELLED;
  }

  RNA_string_get(op->ptr, "filepath", job->filepath);
  if (job->filepath[0] == '\0') {
    const char *blend_path = BKE_main_blendfile_path_from_global();
    if (blend_path[0] != '\0') {
      BLI_snprintf(job->filepath, sizeof(job->filepath), "//%s.nc2", mesh->id.name + 2);
      BLI_path_abs(job->filepath, blend_path);
    }
    else {
      char cwd[FILE_MAX];
      BLI_current_working_dir(cwd, sizeof(cwd));
      char nc2_name[FILE_MAX];
      BLI_snprintf(nc2_name, sizeof(nc2_name), "%s.nc2", mesh->id.name + 2);
      BLI_path_join(job->filepath, sizeof(job->filepath), cwd, nc2_name);
    }
  }

  BLI_strncpy(job->stored_path, job->filepath, sizeof(job->stored_path));
  const char *blend_path = BKE_main_blendfile_path_from_global();
  if (blend_path[0] != '\0') {
    BLI_path_rel(job->stored_path, blend_path);
  }

  /* Tag object as actively building clusters for UI reactivity. */
  if (IDProperty *idgroup = IDP_EnsureProperties(&ob->id)) {
    idp_int_replace(idgroup, "tessera_building", 1);
    idp_float_replace(idgroup, "tessera_build_progress", 0.0f);
    idp_string_replace(idgroup, "tessera_build_status", "Starting...");
  }
  WM_main_add_notifier(NC_OBJECT | ND_DRAW, ob);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_VIEW3D, nullptr);

  if (is_invoke) {
    job->is_nonblocking_job = true;
    op->customdata = job;
    wmWindowManager *wm = CTX_wm_manager(C);
    wmWindow *win = CTX_wm_window(C);

    wmJob *wm_job = WM_jobs_get(wm,
                                win,
                                scene,
                                "Building Tessera Clusters",
                                WM_JOB_PROGRESS,
                                WM_JOB_TYPE_OBJECT_BAKE);
    job->wm_job = wm_job;

    WM_jobs_customdata_set(wm_job, job, tessera_build_free_job);
    WM_jobs_timer(wm_job, 0.05, NC_WM | ND_JOB, 0);
    WM_jobs_callbacks(wm_job, tessera_build_start_job, nullptr, nullptr, tessera_build_end_job);

    WM_jobs_start(wm, wm_job);
    WM_event_add_modal_handler(C, op);
    return OPERATOR_RUNNING_MODAL;
  }
  else {
    job->is_nonblocking_job = false;
    wmJobWorkerStatus worker_status = {};
    worker_status.reports = op->reports;
    tessera_build_start_job(job, &worker_status);
    tessera_build_end_job(job);
    const bool success = job->success;
    tessera_build_free_job(job);
    return success ? OPERATOR_FINISHED : OPERATOR_CANCELLED;
  }
}

static wmOperatorStatus object_tessera_build_exec(bContext *C, wmOperator *op)
{
  const bool is_invoke = (op->flag & OP_IS_INVOKE) != 0;
  return object_tessera_build_custom(C, op, is_invoke);
}

static wmOperatorStatus object_tessera_build_invoke(bContext *C, wmOperator *op, const wmEvent * /*event*/)
{
  return object_tessera_build_custom(C, op, true);
}

void OBJECT_OT_tessera_build(wmOperatorType *ot)
{
  ot->name = "Build Tessera Clusters";
  ot->idname = "OBJECT_OT_tessera_build";
  ot->description = "Build the Tessera cluster DAG (QEM simplification) and save "
                    "it to a .nc2 file";

  ot->invoke = object_tessera_build_invoke;
  ot->exec = object_tessera_build_exec;
  ot->modal = object_tessera_build_modal;
  ot->poll = object_tessera_build_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_string_file_path(ot->srna,
                           "filepath",
                           nullptr,
                           FILE_MAX,
                           "File Path",
                           "Output .nc2 file path");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Tessera GPU Test Operator
 *
 * Headless end-to-end validation of the whole pipeline on the real GPU:
 * compute-shader traversal, visibility-buffer rasterization, material
 * shading (all three modes) and async page streaming.
 * \{ */

namespace {

void tessera_test_make_viewproj(float cam_z, float4x4 &r_vp)
{
  /* Perspective, 90 deg vertical fov, near=0.1, far=1000, camera on +Z
   * looking at the origin (column-major Blender matrix). */
  const float f = 1.0f;
  const float n = 0.1f, fa = 1000.0f;
  r_vp = float4x4::identity();
  r_vp[0][0] = f;
  r_vp[1][1] = f;
  r_vp[2][2] = (fa + n) / (n - fa);
  r_vp[3][2] = 2.0f * fa * n / (n - fa);
  r_vp[2][3] = -1.0f;
  r_vp[3][3] = 0.0f;
  /* View is a pure translation (0, 0, -cam_z). */
  r_vp[3][0] = 0.0f;
  r_vp[3][1] = 0.0f;
  r_vp[3][2] = r_vp[2][2] * (-cam_z) + r_vp[3][2];
  r_vp[3][3] = r_vp[2][3] * (-cam_z) + r_vp[3][3];
}

struct TesseraTestTargets {
  gpu::Texture *color = nullptr;
  gpu::Texture *depth = nullptr;
  gpu::FrameBuffer *fb = nullptr;

  bool ensure(int w, int h)
  {
    if (fb != nullptr) {
      return true;
    }
    color = GPU_texture_create_2d("tessera_test_color",
                                  w,
                                  h,
                                  1,
                                  gpu::TextureFormat::SFLOAT_16_16_16_16,
                                  eGPUTextureUsage(GPU_TEXTURE_USAGE_ATTACHMENT |
                                                   GPU_TEXTURE_USAGE_HOST_READ),
                                  nullptr);
    depth = GPU_texture_create_2d("tessera_test_depth",
                                  w,
                                  h,
                                  1,
                                  gpu::TextureFormat::SFLOAT_32_DEPTH,
                                  eGPUTextureUsage(GPU_TEXTURE_USAGE_ATTACHMENT),
                                  nullptr);
    if (color == nullptr || depth == nullptr) {
      return false;
    }
    fb = GPU_framebuffer_create("tessera_test_fb");
    GPU_framebuffer_ensure_config(
        &fb,
        {GPU_ATTACHMENT_TEXTURE(depth), GPU_ATTACHMENT_TEXTURE(color)});
    return fb != nullptr;
  }

  void free_targets()
  {
    GPU_FRAMEBUFFER_FREE_SAFE(fb);
    GPU_TEXTURE_FREE_SAFE(color);
    GPU_TEXTURE_FREE_SAFE(depth);
  }
};

/* Count lit pixels (background is exactly (0,0,0)). */
void tessera_test_read_coverage(TesseraTestTargets &targets,
                               int w,
                               int h,
                               int &r_lit,
                               float r_mean_color[3])
{
  r_lit = 0;
  r_mean_color[0] = r_mean_color[1] = r_mean_color[2] = 0.0f;
  float *pixels = static_cast<float *>(
      GPU_texture_read(targets.color, GPU_DATA_FLOAT, 0));
  if (pixels == nullptr) {
    return;
  }
  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float *px = pixels + (size_t(y) * w + x) * 4;
      if (px[0] != 0.0f || px[1] != 0.0f || px[2] != 0.0f) {
        r_lit++;
        sum[0] += px[0];
        sum[1] += px[1];
        sum[2] += px[2];
      }
    }
  }
  if (r_lit > 0) {
    r_mean_color[0] = sum[0] / r_lit;
    r_mean_color[1] = sum[1] / r_lit;
    r_mean_color[2] = sum[2] / r_lit;
  }
  MEM_delete(pixels);
}

}  // namespace

static bool object_tessera_gpu_test_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  return ob != nullptr && ob->type == OB_MESH && ob->data != nullptr;
}

static wmOperatorStatus object_tessera_gpu_test_exec(bContext *C, wmOperator *op)
{
  if (GPU_context_active_get() == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               "Tessera GPU test needs an active GPU context (call "
               "gpu.init() from Python first)");
    return OPERATOR_CANCELLED;
  }

  Object *ob = CTX_data_active_object(C);
  Mesh *mesh = id_cast<Mesh *>(ob->data);

  tessera::TesseraData data;
  if (!tessera::buildFromMesh(*mesh, data)) {
    BKE_report(op->reports, RPT_ERROR, "Tessera: failed to build cluster DAG");
    return OPERATOR_CANCELLED;
  }

  const int W = 256, H = 256;
  float4x4 vp;
  tessera_test_make_viewproj(3.0f, vp);

  TesseraTestTargets targets;
  if (!targets.ensure(W, H)) {
    BKE_report(
        op->reports, RPT_ERROR, "Tessera GPU test: offscreen setup failed");
    return OPERATOR_CANCELLED;
  }

  tessera::TesseraDrawPass pass(data, vp, 1.0f, float(H));
  if (!pass.init()) {
    BKE_report(
        op->reports, RPT_ERROR, "Tessera GPU test: renderer init failed");
    targets.free_targets();
    return OPERATOR_CANCELLED;
  }

  /* ---- 1. Static rendering in all shading modes ---- */
  int lit[3] = {0, 0, 0};
  float mean[3][3] = {};
  const char *mode_names[3] = {"diffuse", "clusters", "normals"};
  for (int mode = 0; mode < 3; ++mode) {
    pass.setShadingMode(tessera::TesseraShadingMode(mode));
    pass.update(data, vp, 1.0f, float(H));
    GPU_framebuffer_bind(targets.fb);
    GPU_framebuffer_clear_color_depth(targets.fb, double4(0.0, 0.0, 0.0, 1.0), 1.0f);
    pass.draw(W, H);
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);
    tessera_test_read_coverage(targets, W, H, lit[mode], mean[mode]);
    std::fprintf(stderr, "tessera-gpu-test: mode=%s clusters=%zu lit=%d "
                "mean=(%.3f,%.3f,%.3f)\n",
                mode_names[mode],
                pass.drawCount(),
                lit[mode],
                double(mean[mode][0]),
                double(mean[mode][1]),
                double(mean[mode][2]));
  }
  std::fflush(stdout);

  BKE_reportf(op->reports,
              RPT_INFO,
              "Tessera GPU static pass: %zu clusters drawn; lit pixels "
              "diffuse=%d clusters=%d normals=%d",
              pass.drawCount(),
              lit[0],
              lit[1],
              lit[2]);

  /* ---- 2. Streaming render: paged file, async loads, growing coverage ----
   */
  const std::string paged_path = "tessera_gpu_test_paged.nc2";
  bool streaming_ok = false;
  std::vector<uint8_t> paged_blob;
  if (tessera::serializePagedToMemory(
          data, tessera::kTesseraClustersPerPage, paged_blob))
  {
    FILE *file = std::fopen(paged_path.c_str(), "wb");
    if (file != nullptr) {
      std::fwrite(paged_blob.data(), 1, paged_blob.size(), file);
      std::fclose(file);
    }
    if (pass.openStreaming(paged_path, data)) {
      pass.setShadingMode(tessera::TESSERA_SHADING_DIFFUSE);
      size_t first_count = 0;
      for (int frame = 0; frame < 300 && !pass.streamingSettled(); ++frame) {
    pass.update(data, vp, 1.0f, float(H));
        GPU_framebuffer_bind(targets.fb);
        GPU_framebuffer_clear_color_depth(targets.fb, double4(0.0, 0.0, 0.0, 1.0), 1.0f);
    pass.draw(W, H);
        if (frame == 0) {
          first_count = pass.drawCount();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      const size_t last_count = pass.drawCount();
      streaming_ok = pass.streamingSettled() && last_count > 0;
      BKE_reportf(op->reports,
                  RPT_INFO,
                  "Tessera GPU streaming: first frame %zu clusters, "
                  "settled at %zu clusters",
                  first_count,
                  last_count);
      std::fprintf(stderr, "tessera-gpu-test: streaming first=%zu settled=%zu ok=%d\n",
                  first_count,
                  last_count,
                  int(streaming_ok));
      std::fflush(stdout);
      pass.closeStreaming();
      std::remove(paged_path.c_str());
    }
  }
  else {
    BKE_report(op->reports, RPT_WARNING, "Tessera: paged serialization failed");
  }

  /* Unbind our framebuffer before freeing it: the material pass rebound it
   * as the "previous" framebuffer, and deleting a bound FBO crashes some
   * drivers. */
  GPU_framebuffer_restore();
  targets.free_targets();

  if (lit[0] == 0 || lit[1] == 0) {
    BKE_report(
        op->reports, RPT_ERROR, "Tessera GPU test FAILED: nothing rendered");
    return OPERATOR_CANCELLED;
  }
  if (!streaming_ok) {
    BKE_report(
        op->reports, RPT_ERROR, "Tessera GPU test FAILED: streaming broke");
    return OPERATOR_CANCELLED;
  }
  BKE_report(
      op->reports, RPT_INFO, "Tessera GPU test PASSED (all modes + streaming)");
  return OPERATOR_FINISHED;
}

void OBJECT_OT_tessera_gpu_test(wmOperatorType *ot)
{
  ot->name = "Tessera GPU Test";
  ot->idname = "OBJECT_OT_tessera_gpu_test";
  ot->description = "Render the Tessera pipeline offscreen and report coverage "
                    "(traversal + visibility buffer + streaming)";

  ot->exec = object_tessera_gpu_test_exec;
  ot->poll = object_tessera_gpu_test_poll;

  ot->flag = 0; /* test-only operator: no undo, no registration */
}

/** \} */

}  // namespace blender::ed::object
