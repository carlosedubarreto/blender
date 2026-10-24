"""Tessera rendering integration.

This module is a thin Python wrapper around a renderer object (today the
pure-Python data path; the compiled ``blender::tessera`` backend exposes the
same interface). The Python API exposes:

    tessera.render.draw_tessera(data, view_proj, viewport_height, batch, renderer)

where ``data`` is the :class:`TesseraData` returned by
``tessera.load.get_loaded_data()`` and ``batch`` is a ``gpu.types.GPUBatch``
configured with the mesh vertex/index buffers.

Example::

    import bpy
    import tessera

    data = tessera.load.get_loaded_data()
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
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)


def _tag_tessera_redraw(self, context):
    """Propagate a Tessera control change to the render engines.

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
    self.id_data["tessera_enabled"] = int(self.enabled)
    _tag_tessera_redraw(self, context)


def _visualize_clusters_update(self, context):
    self.id_data["tessera_visualize_clusters"] = int(self.visualize_clusters)
    _tag_tessera_redraw(self, context)


def _pixel_error_update(self, context):
    self.id_data["tessera_pixel_error"] = float(self.pixel_error)
    _tag_tessera_redraw(self, context)


def _view_source_update(self, context):
    self.id_data["tessera_view_source"] = int(self.view_source == 'CAMERA')
    _tag_tessera_redraw(self, context)


def draw_tessera(data, view_proj, viewport_height, batch, renderer):
    """Draw the loaded Tessera data for the current frame.

    Returns the number of clusters drawn. Returns 0 if nothing was drawn.
    """
    if renderer is None or batch is None:
        return 0

    renderer.update(data, view_proj, viewport_height)
    if renderer.drawCount() == 0:
        return 0

    renderer.draw(batch)
    return renderer.drawCount()


class TesseraProperties(bpy.types.PropertyGroup):
    """Object-level Tessera properties (mirrored to IDProperties read by the
    C++ draw engine; see the ``update`` callbacks above)."""

    # Master switch: false renders the object through the regular mesh path.
    enabled: BoolProperty(
        name="Use Tessera",
        description="Render this object with Tessera virtual geometry "
                    "(Workbench engine only)",
        default=True,
        update=_enabled_update,
    )

    # Path to the .nc2 file produced by the build operator.
    nc2_path: StringProperty(
        name="Tessera .nc2",
        description="Path to the Tessera cluster file",
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

    # LOD view source: which view drives the screen-space error test.
    view_source: EnumProperty(
        name="View Source",
        description="View used for Tessera LOD selection (screen-space error)",
        items=(
            ('VIEWPORT', "Viewport View",
             "Select LODs from the current viewport view"),
            ('CAMERA', "Active Camera",
             "Select LODs from the scene's active camera"),
        ),
        default='VIEWPORT',
        update=_view_source_update,
    )

    # Toggle cluster visualization mode.
    visualize_clusters: BoolProperty(
        name="Visualize Clusters",
        description="Show each cluster with a flat colour",
        default=False,
        update=_visualize_clusters_update,
    )


def register():
    bpy.utils.register_class(TesseraProperties)
    bpy.types.Object.tessera = PointerProperty(type=TesseraProperties)


def unregister():
    del bpy.types.Object.tessera
    bpy.utils.unregister_class(TesseraProperties)
