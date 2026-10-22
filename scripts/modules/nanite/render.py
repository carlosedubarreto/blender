"""Nanite rendering integration.

This module is a thin Python wrapper around a renderer object (today the
pure-Python data path; the compiled ``blender::nanite`` backend exposes the
same interface). The Python API exposes:

    nanite.render.draw_nanite(data, view_proj, viewport_height, batch, renderer)

where ``data`` is the :class:`NaniteData` returned by
``nanite.load.get_loaded_data()`` and ``batch`` is a ``gpu.types.GPUBatch``
configured with the mesh vertex/index buffers.

Example::

    import bpy
    import nanite

    data = nanite.load.get_loaded_data()
    view_proj = ...            # compute from the current camera
    viewport_height = ...      # pixels
    batch = ...                # GPUBatch for the mesh
    renderer = ...             # object with init/update/draw/drawCount
    renderer.init()
    renderer.update(data, view_proj, viewport_height)
    renderer.draw(batch)
    renderer.terminate()
"""

import bpy
from bpy.props import PointerProperty, StringProperty, BoolProperty, FloatProperty


def _tag_nanite_redraw(self, context):
    """Propagate a Nanite control change to the render engines.

    Editing a property alone does not re-sync the Workbench engine (its cached
    passes would keep the old state), so tag the object for a depsgraph
    re-evaluation (a bare ``update_tag()`` with no refresh flags is a no-op)
    and request a redraw of every 3D viewport.
    """
    self.id_data.update_tag(refresh={'OBJECT'})
    window = context.window
    if window is None or window.screen is None:
        return
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _enabled_update(self, context):
    self.id_data["nanite_enabled"] = int(self.enabled)
    _tag_nanite_redraw(self, context)


def _visualize_clusters_update(self, context):
    self.id_data["nanite_visualize_clusters"] = int(self.visualize_clusters)
    _tag_nanite_redraw(self, context)


def _pixel_error_update(self, context):
    self.id_data["nanite_pixel_error"] = float(self.pixel_error)
    _tag_nanite_redraw(self, context)


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
    """Object-level Nanite properties (mirrored to IDProperties read by the
    C++ draw engine; see the ``update`` callbacks above)."""

    # Master switch: false renders the object through the regular mesh path.
    enabled: BoolProperty(
        name="Use Nanite",
        description="Render this object with Nanite virtual geometry "
                    "(Workbench engine only)",
        default=True,
        update=_enabled_update,
    )

    # Path to the .nc2 file produced by the build operator.
    nc2_path: StringProperty(
        name="Nanite .nc2",
        description="Path to the Nanite cluster file",
        default="",
    )

    # LOD quality: tolerated screen-space error in pixels.
    pixel_error: FloatProperty(
        name="Pixel Error",
        description="Maximum tolerated screen-space error in pixels; "
                    "higher values select coarser clusters (faster, less detail)",
        default=1.0,
        min=0.01,
        soft_min=0.1,
        soft_max=10.0,
        precision=2,
        update=_pixel_error_update,
    )

    # Toggle cluster visualization mode.
    visualize_clusters: BoolProperty(
        name="Visualize Clusters",
        description="Show each cluster with a flat colour",
        default=False,
        update=_visualize_clusters_update,
    )


def register():
    bpy.utils.register_class(NaniteProperties)
    bpy.types.Object.nanite = PointerProperty(type=NaniteProperties)


def unregister():
    del bpy.types.Object.nanite
    bpy.utils.unregister_class(NaniteProperties)
