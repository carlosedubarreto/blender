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
    if "tessera_culling_margin" not in self.id_data:
        self.id_data["tessera_culling_margin"] = float(getattr(self, "culling_margin", 0.02))
    if "tessera_view_source" not in self.id_data:
        self.id_data["tessera_view_source"] = int(getattr(self, "use_camera_view", False))
    if "tessera_texture_lod_mode" not in self.id_data:
        self.id_data["tessera_texture_lod_mode"] = str(getattr(self, "texture_lod_mode", "AUTO"))
    if self.enabled:
        import os
        nc2 = self.id_data.get("tessera_nc2")
        from tessera.ui import get_existing_nc2_path
        existing = get_existing_nc2_path(self.id_data)
        if existing and (not nc2 or not os.path.isfile(bpy.path.abspath(nc2))):
            self.id_data["tessera_nc2"] = existing
    _tag_tessera_redraw(self, context)



def _texture_lod_update(self, context):
    self.id_data["tessera_texture_lod_mode"] = str(self.texture_lod_mode)
    _tag_tessera_redraw(self, context)


def _visualize_clusters_update(self, context):
    self.id_data["tessera_visualize_clusters"] = int(self.visualize_clusters)
    _tag_tessera_redraw(self, context)


def _pixel_error_update(self, context):
    self.id_data["tessera_pixel_error"] = float(self.pixel_error)
    _tag_tessera_redraw(self, context)


def _culling_margin_update(self, context):
    self.id_data["tessera_culling_margin"] = float(self.culling_margin)
    _tag_tessera_redraw(self, context)


def get_effective_margin(ob, context=None):
    if ob is None:
        return 0.02
    return max(0.0, float(ob.get("tessera_culling_margin", 0.02)))


def _view_source_update(self, context):
    val = (self.view_source == 'CAMERA')
    self.id_data["tessera_view_source"] = int(val)
    if self.use_camera_view != val:
        self.use_camera_view = val
    _tag_tessera_redraw(self, context)


def _use_camera_view_update(self, context):
    val = self.use_camera_view
    self.id_data["tessera_view_source"] = int(val)
    target_enum = 'CAMERA' if val else 'VIEWPORT'
    if self.view_source != target_enum:
        self.view_source = target_enum
    _tag_tessera_redraw(self, context)


def _show_stats_update(self, context):
    self.id_data["tessera_show_stats"] = int(self.show_stats)
    _tag_tessera_redraw(self, context)


def _log_console_update(self, context):
    self.id_data["tessera_log_console"] = int(self.log_console)
    _tag_tessera_redraw(self, context)


def _show_fps_panel_update(self, context):
    self.id_data["tessera_show_fps_panel"] = int(self.show_fps_panel)
    _tag_tessera_redraw(self, context)


def _show_fps_hud_update(self, context):
    self.id_data["tessera_show_fps_hud"] = int(self.show_fps_hud)
    try:
        from tessera.ui import update_hud_state
        update_hud_state()
    except Exception:
        pass
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
                    "(Workbench and EEVEE engines)",
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

    # Frustum culling margin: distance in units to keep clusters active beyond camera borders.
    culling_margin: FloatProperty(
        name="Camera Margin",
        description="Distance in units to keep clusters active beyond camera view borders "
                    "(prevents edge popping during panning or off-screen shadows)",
        default=0.02,
        min=0.0,
        soft_max=10.0,
        unit='LENGTH',
        update=_culling_margin_update,
    )

    # Camera LOD toggle: True locks LOD and culling to the active scene camera.
    use_camera_view: BoolProperty(
        name="Use Camera View",
        description="Lock cluster LOD and culling calculation to the active scene camera",
        default=False,
        update=_use_camera_view_update,
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

    # Texture resolution scaling mode.
    texture_lod_mode: EnumProperty(
        name="Texture LOD",
        description="Texture resolution scaling mode",
        items=[
            ('AUTO', "Auto (Distance)", "Dynamically scale texture resolution based on distance to camera"),
            ('FULL', "Full (100%)", "Always use full resolution"),
            ('HALF', "Half (50%)", "Clamp maximum resolution to half"),
            ('QUARTER', "Quarter (25%)", "Clamp maximum resolution to quarter"),
            ('EIGHTH', "Eighth (12.5%)", "Clamp maximum resolution to 1/8th for low VRAM"),
        ],
        default='AUTO',
        update=_texture_lod_update,
    )

    # Toggle cluster visualization mode.
    visualize_clusters: BoolProperty(
        name="Visualize Clusters",
        description="Show each cluster with a flat colour",
        default=False,
        update=_visualize_clusters_update,
    )

    # Cluster build progress (0.0 to 100.0)
    build_progress: FloatProperty(
        name="Progress",
        description="Cluster generation progress",
        subtype='PERCENTAGE',
        min=0.0,
        max=100.0,
        default=0.0,
    )

    # Real-time telemetry toggles
    show_stats: BoolProperty(
        name="Show Live Stats",
        description="Display real-time cluster, triangle, and LOD telemetry in the panel",
        default=True,
        update=_show_stats_update,
    )

    log_console: BoolProperty(
        name="Log Stats to Console",
        description="Print real-time draw metrics to the system console",
        default=False,
        update=_log_console_update,
    )

    show_fps_panel: BoolProperty(
        name="Show FPS in Panel",
        description="Display frame rate and frame time in the sidebar panel",
        default=False,
        update=_show_fps_panel_update,
    )

    show_fps_hud: BoolProperty(
        name="Viewport HUD",
        description="Render a live FPS and frame time overlay directly in the 3D Viewport",
        default=False,
        update=_show_fps_hud_update,
    )



def register():
    bpy.utils.register_class(TesseraProperties)
    bpy.types.Object.tessera = PointerProperty(type=TesseraProperties)


def unregister():
    del bpy.types.Object.tessera
    bpy.utils.unregister_class(TesseraProperties)
