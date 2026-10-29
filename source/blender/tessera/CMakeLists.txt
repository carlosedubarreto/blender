# SPDX-FileCopyrightText: 2006 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Tessera: virtual geometry LOD clusters (Blender side)
#
# This library holds the cluster data structures and LOD selection logic. The
# actual GPU draw call is performed by TesseraRenderer using Blender's GPU batch
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
  ${SRC_DIR}/../depsgraph
  ${SRC_DIR}/../functions
  ${SRC_DIR}/../makesdna
)

set(INC_SYS
)

set(SRC
  tessera.h
  tessera_archetype.hh
  tessera_archetype.cc
  tessera_cluster.hh
  tessera_cluster.cc
  tessera_cluster_serial.cc
  tessera_cluster_builder.hh
  tessera_cluster_builder.cc
  tessera_qem.hh
  tessera_qem.cc
  tessera_lod.hh
  tessera_lod.cc
  tessera_gpu_data.hh
  tessera_gpu_data.cc
  tessera_gpu_traverse.hh
  tessera_gpu_traverse.cc
  tessera_visibility.hh
  tessera_visibility.cc
  tessera_paging.hh
  tessera_paging.cc
  tessera_renderer.hh
  tessera_renderer.cc
  tessera_draw.hh
  tessera_draw.cc
  tessera_module.cc
)

set(LIB
  PRIVATE bf_blenkernel
  PRIVATE bf_blenlib
  PRIVATE bf_depsgraph
  PRIVATE bf_gpu
  PRIVATE bf::blenlib
  PRIVATE bf::dna
  PRIVATE bf::intern::guardedalloc
)

if(WITH_OPENGL_BACKEND)
  add_definitions(-DWITH_OPENGL_BACKEND)
  list(APPEND LIB PRIVATE bf::dependencies::epoxy)
endif()

blender_add_lib(bf_tessera "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
add_library(bf::tessera ALIAS bf_tessera)

if(MSVC AND NOT MSVC_CLANG)
  target_compile_options(bf_tessera PRIVATE /WX /wd4101)
endif()
