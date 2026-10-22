# SPDX-FileCopyrightText: 2006 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Nanite: virtual geometry LOD clusters (Blender side)
#
# This library holds the cluster data structures and LOD selection logic. The
# actual GPU draw call is performed by NaniteRenderer using Blender's GPU batch
# + indirect draw API.

set(INC
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  ../../gpu
  ../../gpu/vulkan
  ../draw/intern
  ../../blenkernel
  ../../blenlib
  ../../functions
)

set(INC_SYS
)

set(SRC
  nanite.h
  nanite_cluster.hh
  nanite_cluster.cc
  nanite_lod.hh
  nanite_lod.cc
  nanite_renderer.hh
  nanite_renderer.cc
  nanite_draw.hh
  nanite_draw.cc
  nanite_module.cc
)

set(LIB
  PRIVATE bf_math
  PRIVATE bf_blender
  PRIVATE bf_gpu
)

blender_add_lib(bf_nanite "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
add_library(bf::nanite ALIAS bf_nanite)

if(MSVC AND NOT MSVC_CLANG)
  target_compile_options(bf_nanite PRIVATE /WX /wd4101)
endif()
