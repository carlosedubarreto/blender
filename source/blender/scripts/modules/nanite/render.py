"""Nanite rendering integration.

This module is a thin Python wrapper around the compiled GPU backend
(``blender::nanite``). The Python API exposes:

    nanite.render.draw_nanite(data, view_proj, viewport_height, batch, renderer)

where ``renderer`` is a :class:`NaniteRenderer` instance and ``batch`` is a
``gpu::Batch`` configured with the mesh vertex/index buffers.

Example::

    import bpy
    import nanite

    data = bpy.context.window_manager.nanite_data
    view_proj = ...            # compute from the current camera
    viewport_height = ...      # pixels
    batch = ...                # gpu::Batch for the mesh
    renderer = nanite.NaniteRenderer()
    renderer.init()
    renderer.update(data, view_proj, viewport_height)
    renderer.draw(batch)
    renderer.terminate()
"""

import bpy
from bpy.props import PointerProperty

from ..nanite import NaniteRenderer  # C++ renderer class


def draw_nanite(data, view_proj, viewport_height, batch, renderer):
    """Draw the loaded Nanite data for the current frame.

    Returns the number of clusters drawn. Returns 0 if nothing was drawn.
    """
    if renderer is None or batch is None:
        return 0

    renderer.update(data, view_proj, viewport_height)
    if renderer.drawCount() == 0:
        return 0

    renderer.draw(batch)
    return renderer.drawCount()


class NaniteProperties(bpy.types.PropertyGroup):
    """Object-level Nanite properties."""

    # Path to the .nc2 file produced by the build operator.
    nc2_path: StringProperty(
        name="Nanite .nc2",
        description="Path to the Nanite cluster file",
        default="",
    )

    # Toggle cluster visualization mode.
    visualize_clusters: BoolProperty(
        name="Visualize Clusters",
        description="Show each cluster with a flat colour",
        default=False,
    )


def register():
    bpy.types.Object.nanite = PointerProperty(type=NaniteProperties)


def unregister():
    del bpy.types.Object.nanite
