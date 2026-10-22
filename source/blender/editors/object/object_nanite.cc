/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 * \ingroup edobj
 *
 * Nanite build operator: the C<->Python bridge.
 *
 * `bpy.ops.nanite.build` (scripts/modules/nanite/build.py) delegates here so
 * the real C++ cluster DAG builder (QEM simplification) runs instead of the
 * Python placeholder.
 */

#include <cstdio>
#include <vector>

#include "MEM_guardedalloc.h"

#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_path_utils.hh"
#include "BLI_fileops.hh"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "BKE_context.hh"
#include "BKE_global.hh"
#include "BKE_main.hh"
#include "BKE_report.hh"

#include "ED_object.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "nanite.h"

#include "object_intern.hh"

namespace blender::ed::object {

/* -------------------------------------------------------------------- */
/** \name Nanite Build Operator
 * \{ */

static bool object_nanite_build_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  return ob != nullptr && ob->type == OB_MESH && ob->data != nullptr;
}

static wmOperatorStatus object_nanite_build_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  Mesh *mesh = id_cast<Mesh *>(ob->data);

  nanite::NaniteData data;
  if (!nanite::buildFromMesh(*mesh, data)) {
    BKE_report(op->reports, RPT_ERROR, "Nanite: failed to build cluster DAG from mesh");
    return OPERATOR_CANCELLED;
  }

  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  if (filepath[0] == '\0') {
    BLI_snprintf(filepath, sizeof(filepath), "//%s.nc2", mesh->id.name + 2);
    BLI_path_abs(filepath, BKE_main_blendfile_path_from_global());
  }

  std::vector<uint8_t> blob;
  if (!nanite::serializeToMemory(data, blob)) {
    BKE_report(op->reports, RPT_ERROR, "Nanite: failed to serialize cluster data");
    return OPERATOR_CANCELLED;
  }

  FILE *file = BLI_fopen(filepath, "wb");
  if (file == nullptr) {
    BKE_reportf(
        op->reports, RPT_ERROR, "Nanite: could not open '%s' for writing", filepath);
    return OPERATOR_CANCELLED;
  }
  const size_t written = fwrite(blob.data(), 1, blob.size(), file);
  fclose(file);
  if (written != blob.size()) {
    BKE_reportf(op->reports, RPT_ERROR, "Nanite: short write to '%s'", filepath);
    return OPERATOR_CANCELLED;
  }

  BKE_reportf(op->reports,
              RPT_INFO,
              "Nanite: built %u clusters, %u levels (%zu bytes) -> %s",
              uint(data.clusters.size()),
              data.header.levelCount,
              blob.size(),
              filepath);
  return OPERATOR_FINISHED;
}

void OBJECT_OT_nanite_build(wmOperatorType *ot)
{
  ot->name = "Build Nanite Clusters";
  ot->idname = "OBJECT_OT_nanite_build";
  ot->description = "Build the Nanite cluster DAG (QEM simplification) and save "
                    "it to a .nc2 file";

  ot->exec = object_nanite_build_exec;
  ot->poll = object_nanite_build_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_string_file_path(ot->srna,
                           "filepath",
                           nullptr,
                           FILE_MAX,
                           "File Path",
                           "Output .nc2 file path");
}

/** \} */

}  // namespace blender::ed::object
