path = 'source/blender/draw/engines/workbench/workbench_engine.cc'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add tessera_min_screen_radius_get
target1 = """  static float tessera_lod_intensity_get(const Scene *scene, Object *ob)
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
  }"""

replacement1 = target1 + """

  static float tessera_min_screen_radius_get(const Scene *scene, Object *ob)
  {
    if (ob != nullptr) {
      const IDProperty *prop = tessera_property_find_safe(ob, "tessera_min_screen_radius");
      if (prop != nullptr) {
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
        return (value >= 0.0f) ? value : 0.0f;
      }
    }
    if (scene != nullptr) {
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_batch_min_screen_radius")) {
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
        return (value >= 0.0f) ? value : 0.0f;
      }
      if (const IDProperty *prop = tessera_scene_property_find_safe(scene, "tessera_min_screen_radius")) {
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
        return (value >= 0.0f) ? value : 0.0f;
      }
    }
    return 1.0f;
  }"""

assert target1 in content, "target1 not found"
content = content.replace(target1, replacement1, 1)

# 2. Update the banding and add sub-pixel culling
target2 = """        state.frame_band_count = 0;
        for (std::vector<float4x4> &band : state.frame_band_instances) {
          band.clear();
        }
        std::array<float, 8> band_nearest_d;
        std::array<float4x4, 8> band_nearest_mat;
        band_nearest_d.fill(std::numeric_limits<float>::max());
        if (!state.frame_visible_instances.empty()) {
          constexpr float kBandRatio = 2.5f;
          constexpr int kMaxBands = 3;
          const float3 band_cam_pos = tessera_cull_view.camPos;
          float dmin = std::numeric_limits<float>::max();
          float dmax = 0.0f;
          for (const float4x4 &inst_mat : state.frame_visible_instances) {
            const float3 c = (inst_mat * float4(field_bs_center, 1.0f)).xyz();
            const float d = math::distance(c, band_cam_pos);
            dmin = std::min(dmin, d);
            dmax = std::max(dmax, d);
          }
          int band_count = 1;
          if (dmax > dmin * 1.25f && dmin > 1e-6f) {
            const float steps = std::log(dmax / dmin) / std::log(kBandRatio);
            band_count = std::clamp(int(std::ceil(steps)), 1, kMaxBands);
          }
          if (band_count <= 1) {
            state.frame_band_instances[0] = state.frame_visible_instances;
            band_nearest_d[0] = dmin;
            if (!state.frame_visible_instances.empty()) {
              band_nearest_mat[0] = state.frame_visible_instances[0];
            }
          }
          else {
            const float t0 = dmin * kBandRatio;
            const float t1 = t0 * kBandRatio;
            for (const float4x4 &inst_mat : state.frame_visible_instances) {
              const float3 c = (inst_mat * float4(field_bs_center, 1.0f)).xyz();
              const float d = math::distance(c, band_cam_pos);
              const int k = (d < t0) ? 0 : ((d < t1) ? 1 : (band_count - 1));
              state.frame_band_instances[k].push_back(inst_mat);
              if (d < band_nearest_d[k]) {
                band_nearest_d[k] = d;
                band_nearest_mat[k] = inst_mat;
              }
            }
          }
          state.frame_band_count = uint32_t(band_count);
        }"""

replacement2 = """        state.frame_band_count = 0;
        for (std::vector<float4x4> &band : state.frame_band_instances) {
          band.clear();
        }
        std::array<float, 8> band_nearest_d;
        std::array<float4x4, 8> band_nearest_mat;
        band_nearest_d.fill(std::numeric_limits<float>::max());
        if (!state.frame_visible_instances.empty()) {
          constexpr float kBandRatio = 1.7f;
          constexpr int kMaxBands = 6;
          const float log_ratio = std::log(kBandRatio);
          const float3 band_cam_pos = tessera_cull_view.camPos;
          const float min_screen_radius_px = tessera_min_screen_radius_get(scene_state_.scene, state.ob);
          const float px_scale = viewport_proj_scale_y * (float(resolution.y) * 0.5f);

          float dmin = std::numeric_limits<float>::max();
          float dmax = 0.0f;
          for (const float4x4 &inst_mat : state.frame_visible_instances) {
            const float3 c = (inst_mat * float4(field_bs_center, 1.0f)).xyz();
            const float d = math::distance(c, band_cam_pos);
            dmin = std::min(dmin, d);
            dmax = std::max(dmax, d);
          }
          int band_count = 1;
          if (dmax > dmin * 1.25f && dmin > 1e-6f) {
            const float steps = std::log(dmax / dmin) / log_ratio;
            band_count = std::clamp(int(std::ceil(steps)), 1, kMaxBands);
          }

          for (const float4x4 &inst_mat : state.frame_visible_instances) {
            const float3 c = (inst_mat * float4(field_bs_center, 1.0f)).xyz();
            const float d = math::distance(c, band_cam_pos);

            /* Sub-pixel instance culling: instances whose projected bounding
             * sphere radius is smaller than min_screen_radius_px are sub-pixel
             * and discarded. */
            if (min_screen_radius_px > 0.0f && have_bs && field_bs_radius > 0.0f) {
              const float inst_scale = std::max(
                  math::length(inst_mat[0].xyz()),
                  std::max(math::length(inst_mat[1].xyz()), math::length(inst_mat[2].xyz())));
              const float world_radius = field_bs_radius * inst_scale;
              const float screen_radius_px = (world_radius * px_scale) / std::max(d, 0.01f);
              if (screen_radius_px < min_screen_radius_px) {
                continue;
              }
            }

            int k = 0;
            if (band_count > 1 && d > dmin && log_ratio > 1e-5f) {
              k = std::clamp(int(std::floor(std::log(std::max(d / dmin, 1.0f)) / log_ratio)),
                             0,
                             band_count - 1);
            }
            state.frame_band_instances[k].push_back(inst_mat);
            if (d < band_nearest_d[k]) {
              band_nearest_d[k] = d;
              band_nearest_mat[k] = inst_mat;
            }
          }
          state.frame_band_count = uint32_t(band_count);
        }"""

assert target2 in content, "target2 not found"
content = content.replace(target2, replacement2, 1)

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("Applied subpixel culling and multi-banding to workbench_engine.cc")

