/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 *  \ingroup tessera
 *  Realized-geometry archetype resolution (see tessera_archetype.hh).
 */

#include "tessera_archetype.hh"

#include "BKE_geometry_set.hh"
#include "BKE_idprop.hh"
#include "BKE_main.hh"
#include "BKE_material.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"

#include "BLI_fileops.hh"
#include "BLI_listbase.h"
#include "BLI_path_utils.hh"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "DEG_depsgraph_query.hh"

#include "DNA_ID.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include <cstdio>
#include <cstring>

namespace blender::tessera {

namespace {

struct MeshFingerprint {
  int64_t verts = -1;
  int64_t corners = -1;
  bool operator==(const MeshFingerprint &other) const
  {
    return verts == other.verts && corners == other.corners;
  }
};

MeshFingerprint mesh_fingerprint(const Mesh &mesh)
{
  MeshFingerprint fp;
  fp.verts = int64_t(mesh.vert_positions().size());
  fp.corners = int64_t(mesh.corner_verts().size());
  return fp;
}

/** Workbench material base color of a mesh or object's first material slot. */
bool mesh_first_material_color(const Mesh &mesh, const Object *ob, float4 &r_color)
{
  if (mesh.mat != nullptr && mesh.totcol >= 1 && mesh.mat[0] != nullptr) {
    const Material *ma = mesh.mat[0];
    r_color = float4(ma->r, ma->g, ma->b, 1.0f);
    return true;
  }
  if (ob != nullptr) {
    Material *ma = BKE_object_material_get(const_cast<Object *>(ob), 1);
    if (ma != nullptr) {
      r_color = float4(ma->r, ma->g, ma->b, 1.0f);
      return true;
    }
  }
  return false;
}

bool object_nc2_idprop_get(const Object *ob, const char *blend_path, std::string &r_path)
{
  if (ob == nullptr) {
    return false;
  }
  const ID *id = &ob->id;
  if (id->properties != nullptr && id->properties->type == IDP_GROUP) {
    if (const IDProperty *prop = IDP_GetPropertyFromGroup(id->properties, "tessera_nc2")) {
      if (prop->type == IDP_STRING) {
        const char *val = IDP_string_get(prop);
        if (val != nullptr && val[0] != '\0') {
          char abs_path[FILE_MAX];
          BLI_strncpy(abs_path, val, sizeof(abs_path));
          if (blend_path != nullptr && blend_path[0] != '\0') {
            BLI_path_abs(abs_path, blend_path);
          }
          r_path = abs_path;
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

ArchetypeResolver &ArchetypeResolver::global()
{
  static ArchetypeResolver resolver;
  return resolver;
}

void ArchetypeResolver::clear()
{
  std::lock_guard lock(mutex_);
  resolved_.clear();
}

TesseraArchetype ArchetypeResolver::resolve(const Main *bmain,
                                            const char *blend_path,
                                            const Object *emitter,
                                            const ID *dupli_ob_data)
{
  TesseraArchetype result;
  const Mesh *realized = nullptr;
  if (dupli_ob_data != nullptr && GS(dupli_ob_data->name) == ID_ME) {
    realized = reinterpret_cast<const Mesh *>(dupli_ob_data);
  }

  /* 1. If the passed dupli_ob_data is missing or is a tiny proxy cube (<= 24 verts),
   * inspect the emitter's evaluated GeometrySet if available. */
  if ((realized == nullptr || mesh_fingerprint(*realized).verts <= 24) && emitter != nullptr &&
      emitter->runtime != nullptr && emitter->runtime->geometry_set_eval != nullptr)
  {
    if (const Mesh *eval_mesh = emitter->runtime->geometry_set_eval->get_mesh()) {
      if (mesh_fingerprint(*eval_mesh).verts > 24) {
        realized = eval_mesh;
      }
    }
  }

  if (realized == nullptr) {
    return result;
  }
  const MeshFingerprint fp = mesh_fingerprint(*realized);
  if (fp.verts <= 0 || fp.corners <= 0) {
    return result;
  }

  std::lock_guard lock(mutex_);

  /* Fast path: previously resolved realized meshes (pointer + fingerprint). */
  if (const CacheEntry *entry = resolved_.lookup_ptr(realized)) {
    if (entry->fp_verts == fp.verts && entry->fp_corners == fp.corners) {
      result = entry->arch;
      result.valid = true;
      return result;
    }
    /* Pointer was recycled for different geometry: drop the stale entry. */
    resolved_.remove(realized);
  }

  /* Match the realized mesh against the original objects in main. Geometry
   * Nodes realization copies the source mesh one-to-one (no merge), so the
   * realized (vertex, corner) counts identify the source plant variety. */
  const Mesh *matched_mesh = nullptr;
  const Object *matched_ob = nullptr;
  if (bmain != nullptr && fp.verts > 24) {
    for (const Object *ob = static_cast<const Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<const Object *>(ob->id.next))
    {
      if (ob->type != OB_MESH || ob->data == nullptr) {
        continue;
      }
      const Mesh *me = reinterpret_cast<const Mesh *>(ob->data);
      if (me == nullptr || me == matched_mesh) {
        continue;
      }
      const MeshFingerprint cand = mesh_fingerprint(*me);
      if (cand == fp) {
        if (matched_mesh != nullptr && matched_mesh != me) {
          /* Ambiguous: two different originals share the fingerprint. */
          matched_mesh = nullptr;
          matched_ob = nullptr;
          break;
        }
        matched_mesh = me;
        matched_ob = ob;
      }
    }
  }

  /* 2. Name / Prefix Matching fallback:
   * When the emitter object is a low-poly proxy cube (e.g. "DewBerry", 8 verts)
   * and no exact fingerprint match was found, match against original scene mesh
   * archetypes matching the emitter name prefix (e.g. "DewBerry_V1", "DewBerry_V2"). */
  if (matched_mesh == nullptr && bmain != nullptr && emitter != nullptr) {
    const char *emitter_name = emitter->id.name + 2;
    const size_t prefix_len = strlen(emitter_name);
    if (prefix_len > 0 && fp.verts <= 24) {
      for (const Object *ob = static_cast<const Object *>(bmain->objects.first); ob != nullptr;
           ob = static_cast<const Object *>(ob->id.next))
      {
        if (ob == emitter || ob->type != OB_MESH || ob->data == nullptr) {
          continue;
        }
        const char *cand_name = ob->id.name + 2;
        if (STREQLEN(cand_name, emitter_name, prefix_len) &&
            (cand_name[prefix_len] == '_' || cand_name[prefix_len] == '.' ||
             cand_name[prefix_len] == ' '))
        {
          const Mesh *me = reinterpret_cast<const Mesh *>(ob->data);
          if (me != nullptr && mesh_fingerprint(*me).verts > 24) {
            matched_mesh = me;
            matched_ob = ob;
            break;
          }
        }
      }
    }
  }

  if (matched_mesh != nullptr && matched_ob != nullptr) {
    /* Found the archetype object (e.g. "DewBerry_V1"): use its pre-built
     * .nc2 cluster DAG so Tessera draws the exact geometry the native
     * pipeline renders. */
    result.valid = true;
    result.source_name = std::string(matched_mesh->id.name + 2);
    result.matched_object_name = std::string(matched_ob->id.name + 2);
    result.two_sided = true;
    float4 col;
    if (mesh_first_material_color(*matched_mesh, matched_ob, col)) {
      result.base_color = col;
      result.base_color_valid = true;
    }

    std::string idprop_nc2;
    if (object_nc2_idprop_get(matched_ob, blend_path, idprop_nc2)) {
      BLI_stat_t info;
      if (BLI_stat(idprop_nc2.c_str(), &info) == 0 && info.st_size > 88) {
        result.nc2_path = idprop_nc2;
        result.nc2_mtime = int64_t(info.st_mtime);
        result.nc2_size = int64_t(info.st_size);
      }
    }

    if (result.nc2_path.empty() && blend_path != nullptr && blend_path[0] != '\0') {
      const char *candidates[] = {
          matched_mesh->id.name + 2, /* <mesh data name>.nc2 (build operator) */
          matched_ob->id.name + 2,   /* <object name>.nc2                    */
      };
      for (const char *base_name : candidates) {
        char path[FILE_MAX];
        BLI_snprintf(path, sizeof(path), "//%s.nc2", base_name);
        BLI_path_abs(path, blend_path);
        BLI_stat_t info;
        if (BLI_stat(path, &info) == 0 && info.st_size > 88) {
          result.nc2_path = path;
          result.nc2_mtime = int64_t(info.st_mtime);
          result.nc2_size = int64_t(info.st_size);
          break;
        }
      }
    }
    printf("Tessera: emitter '%s' realized range matched archetype '%s' (%lld verts)%s\n",
           emitter ? emitter->id.name + 2 : "?",
           result.source_name.c_str(),
           long long(mesh_fingerprint(*matched_mesh).verts),
           result.nc2_path.empty() ? " [no .nc2, building]" : "");
    fflush(stdout);
    if (!result.nc2_path.empty()) {
      CacheEntry entry;
      entry.arch = result;
      entry.fp_verts = fp.verts;
      entry.fp_corners = fp.corners;
      resolved_.add(realized, entry);
      return result;
    }
    /* No pre-built .nc2: build from the matched botanical mesh. */
    realized = matched_mesh;
  }
  {
    TesseraData data;
    if (!buildFromMesh(*realized, data)) {
      return result;
    }
    result.valid = true;
    if (result.source_name.empty()) {
      result.source_name = "realized_" + std::to_string(fp.verts) + "v_" +
                           std::to_string(fp.corners) + "c";
    }
    result.built_data = std::make_shared<const TesseraData>(std::move(data));
    float4 col;
    if (!result.base_color_valid && mesh_first_material_color(*realized, matched_ob, col)) {
      result.base_color = col;
      result.base_color_valid = true;
    }
    /* Realized scatter geometry is almost always alpha-cutout foliage cards. */
    result.two_sided = true;
    printf("Tessera: emitter '%s' realized range built from geometry (%lld verts, "
           "%u clusters)\n",
           emitter ? emitter->id.name + 2 : "?",
           long long(mesh_fingerprint(*realized).verts),
           result.built_data->header.clusterCount);
    fflush(stdout);
    CacheEntry entry;
    entry.arch = result;
    entry.fp_verts = fp.verts;
    entry.fp_corners = fp.corners;
    resolved_.add(realized, entry);
  }
  return result;
}

}  // namespace blender::tessera
