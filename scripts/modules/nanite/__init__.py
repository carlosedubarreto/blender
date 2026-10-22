"""Nanite virtual geometry cluster module.

Provides operators to build, save and load Nanite cluster LOD data for
Blender meshes, plus a Python API that mirrors the C++ library
(blender::nanite).

Example::

    import bpy

    # Build Nanite data for the active mesh and save it to disk.
    bpy.ops.nanite.build('INVOKE_DEFAULT')

    # Later, load the same data back from the .nc2 file.
    bpy.ops.nanite.load('INVOKE_DEFAULT')
"""

import bpy
from bpy.props import StringProperty, BoolProperty

# Register the C++ library symbols (optional: done automatically by Blender
# when the library is linked). We only expose Python functions here.
from . import build, load, render, ui  # noqa: F401

bl_info = {
    "name": "Nanite",
    "author": "Blender + Nanite CUP2 contributors",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "Properties > Object > Nanite",
    "description": "Build and visualize Nanite-style virtual geometry LODs.",
    "warning": "Experimental - core cluster builder and render backend only.",
    "doc_url": "https://github.com/blender/blender",
    "link": "https://github.com/blender/blender",
}


def register():
    build.register()
    load.register()
    render.register()
    ui.register()


def unregister():
    build.unregister()
    load.unregister()
    render.unregister()
    ui.unregister()
