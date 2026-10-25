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

namespace blender::eevee {

class Instance;

class TesseraModule {
 private:
  Instance &inst_;

  struct ObjectState {
    std::string nc2_path;
    int64_t file_mtime = -1;
    int64_t file_size = -1;
    bool used = false;
    int visualize_clusters = 0;
    bool tessera_enabled = true;
    float pixel_error = 1.0f;
    int view_source = 0;
    float4x4 object_mat;
    Object *ob = nullptr;
    bool show_stats = true;
    bool log_console = false;
    tessera::TesseraData data;
    std::vector<tessera::TesseraMaterialTextures> materials;
    std::unique_ptr<tessera::TesseraDrawPass> pass;
  };

  std::mutex mutex_;
  Map<std::string, std::unique_ptr<ObjectState>> objects_;

 public:
  TesseraModule(Instance &inst);
  ~TesseraModule();

  void begin_sync();
  bool record(Object *ob);
  void draw_prepass(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *depth_fb);
  void draw(const draw::View &view, const int2 &resolution, gpu::FrameBuffer *target_fb);
};

}  // namespace blender::eevee
