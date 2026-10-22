# SPDX-FileCopyrightText: 2006 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Nanite: virtual geometry LOD clusters (Blender side)
#
# This library holds the cluster data structures and LOD selection logic. The
# actual GPU draw call is performed by NaniteRenderer using Blender's GPU batch
# + indirect draw API.

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(INC
  PUBLIC ${SRC_DIR}
  ${SRC_DIR}/../gpu
  ${SRC_DIR}/../gpu/intern
  ${SRC_DIR}/../gpu/vulkan
  ${SRC_DIR}/../draw/intern
  ${SRC_DIR}/../blenkernel
  ${SRC_DIR}/../blenlib
  ${SRC_DIR}/../functions
)

set(INC_SYS
)

set(SRC
  nanite.h
  nanite_cluster.hh
  nanite_cluster.cc
  nanite_cluster_serial.cc
  nanite_cluster_builder.hh
  nanite_cluster_builder.cc
  nanite_qem.hh
  nanite_qem.cc
  nanite_lod.hh
  nanite_lod.cc
  nanite_gpu_data.hh
  nanite_gpu_data.cc
  nanite_gpu_traverse.hh
  nanite_gpu_traverse.cc
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
  PRIVATE bf::blenlib
  PRIVATE bf::intern::guardedalloc
)

blender_add_lib(bf_nanite "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
add_library(bf::nanite ALIAS bf_nanite)

if(MSVC AND NOT MSVC_CLANG)
  target_compile_options(bf_nanite PRIVATE /WX /wd4101)
endif()
