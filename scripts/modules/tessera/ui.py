"""Tessera user interface.

Adds a *Tessera* panel to the 3D viewport sidebar (N-panel) exposing the
available Tessera tools: building cluster data from the active mesh, loading
previously saved ``.nc2`` files, and toggling cluster visualization.

The panel is automatically registered together with the rest of the
:mod:`tessera` module by :func:`tessera.register`.
"""

import os

import bpy


def _tessera_sidebar_refresh_timer():
    """Continuous timer keeping sidebar telemetry and progress indicators updated."""
    context = bpy.context
    ob = getattr(context, "active_object", None)

    is_building = False
    try:
        is_building = any(bool(o.get("tessera_building", 0)) for o in bpy.data.objects)
    except Exception:
        pass

    needs_stats = False
    if ob and bool(ob.get("tessera_enabled", False)):
        props = getattr(ob, "tessera", None)
        if props and getattr(props, "show_stats", False):
            needs_stats = True

    # If building or live stats are active, explicitly redraw the sidebar UI region
    if is_building or needs_stats:
        wm = getattr(context, "window_manager", None)
        if wm:
            for window in wm.windows:
                screen = getattr(window, "screen", None)
                if screen:
                    for area in screen.areas:
                        if area.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'UI':
                                    region.tag_redraw()
        return 0.08  # ~12 FPS refresh rate for UI readouts (very lightweight)

    # Recheck periodically at low frequency if inactive
    return 0.5


def register_refresh_timer():
    """Ensure the continuous sidebar refresh timer is registered."""
    if not bpy.app.timers.is_registered(_tessera_sidebar_refresh_timer):
        bpy.app.timers.register(_tessera_sidebar_refresh_timer, persistent=True)


def unregister_refresh_timer():
    """Unregister the continuous sidebar refresh timer."""
    if bpy.app.timers.is_registered(_tessera_sidebar_refresh_timer):
        bpy.app.timers.unregister(_tessera_sidebar_refresh_timer)


def get_existing_nc2_path(ob):
    """Detect whether a valid compiled Tessera .nc2 binary already exists on disk."""
    if ob is None:
        return None

    blend_dir = bpy.path.abspath("//")
    mesh = getattr(ob, "data", None)
    mesh_name = mesh.name if mesh else ""

    candidates = [
        ob.get("tessera_nc2", ""),
        ob.get("tessera_nc2_path", ""),
        os.path.join(blend_dir, f"{ob.name}.nc2"),
        os.path.join(blend_dir, f"{mesh_name}.nc2") if mesh_name else "",
        os.path.join(blend_dir, f"{ob.name.replace('.', '_')}.nc2"),
        os.path.join(blend_dir, f"{mesh_name.replace('.', '_')}.nc2") if mesh_name else "",
        os.path.join(os.getcwd(), f"{ob.name}.nc2"),
        os.path.join(os.getcwd(), f"{mesh_name}.nc2") if mesh_name else "",
    ]

    for path in candidates:
        if not path:
            continue
        abs_p = bpy.path.abspath(path)
        if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 88:
            return abs_p
    return None


class TESSERA_PT_object(bpy.types.Panel):
    """Tessera options for the active mesh object."""

    bl_label = "Tessera"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        tessera_props = getattr(ob, "tessera", None)

        is_building = bool(ob.get("tessera_building", False))
        build_progress = float(ob.get("tessera_build_progress", 0.0))
        status_text = str(ob.get("tessera_build_status", "Processing..."))

        # State 1: Active Background Cluster Build
        if is_building:
            register_refresh_timer()
            box = layout.box()
            if not status_text:
                status_text = "Processing..."
            box.label(text=status_text, icon='TIME')
            p_row = box.row(align=True)
            p_row.enabled = False
            p_row.label(text=f"Progress: {build_progress * 100.0:.1f}%")

            row = layout.row(align=True)
            row.enabled = False
            row.operator("object.tessera_build", text="Building Clusters...", icon='TIME')
            return

        nc2_path = get_existing_nc2_path(ob)
        has_existing = nc2_path is not None
        is_enabled = bool(ob.get("tessera_enabled", False))

        # State 2: Existing Tessera Cache Detected on Disk
        if has_existing:
            box = layout.box()
            box.label(text="Tessera Cache Found on Disk", icon='CHECKMARK')

            try:
                file_size_mb = os.path.getsize(nc2_path) / (1024 * 1024)
                size_str = f"{file_size_mb:.1f} MB"
            except Exception:
                size_str = "Available"
            box.label(text=f"File: {os.path.basename(nc2_path)} ({size_str})", icon='FILE_CACHE')

            if tessera_props is not None:
                row = box.row(align=True)
                toggle_text = "Disable Tessera" if is_enabled else "Enable Tessera"
                toggle_icon = 'CHECKBOX_HLT' if is_enabled else 'CHECKBOX_DEHLT'
                row.prop(tessera_props, "enabled", text=toggle_text, toggle=True, icon=toggle_icon)

            col = layout.column(align=True)
            col.operator("object.tessera_build", text="Rebuild Tessera Cache", icon='FILE_REFRESH')
            col.operator("tessera.clear", text="Clear Tessera Data", icon='X')

            if is_enabled and tessera_props is not None:
                prop_col = layout.column(align=True)
                prop_col.use_property_split = True
                prop_col.use_property_decorate = False
                prop_col.prop(tessera_props, "pixel_error", text="Pixel Error")
                prop_col.prop(tessera_props, "view_source", text="View Source")
                prop_col.prop(tessera_props, "visualize_clusters")

                # Realtime Telemetry Sub-Panel
                t_box = layout.box()
                t_box.label(text="Realtime Telemetry", icon='VIEW_CAMERA')
                t_row = t_box.row(align=True)
                t_row.prop(tessera_props, "show_stats", text="Live Stats", icon='INFO')
                t_row.prop(tessera_props, "log_console", text="Console Log", icon='CONSOLE')

                if tessera_props.show_stats:
                    t_col = t_box.column(align=True)
                    lod_min = ob.get('tessera_stat_lod_min', 0)
                    lod_max = ob.get('tessera_stat_lod_max', 0)
                    t_col.label(text=f"Active LODs: {lod_min} \u2013 {lod_max}", icon='ZOOM_IN')

                    active_clusters = ob.get('tessera_stat_clusters', 0)
                    total_clusters = ob.get('tessera_total_clusters', 0)
                    t_col.label(text=f"Visible Clusters: {active_clusters:,} / {total_clusters:,}", icon='MOD_SOLIDIFY')

                    active_tris = ob.get('tessera_stat_triangles', 0)
                    total_tris = ob.get('tessera_total_triangles', 0)
                    t_col.label(text=f"Rendered Tris: {active_tris:,} / {total_tris:,}", icon='MESH_DATA')

                    dist = ob.get('tessera_stat_distance', 0.0)
                    err = ob.get('tessera_stat_pixel_error', ob.get('tessera_stat_error', 1.0))
                    t_col.label(text=f"Camera Distance: {dist:.2f} m", icon='CAMERA_DATA')
                    t_col.label(text=f"Projected Error: {err:.1f} px", icon='DRIVER_DISTANCE')

        # State 3: No Tessera Cache Found
        else:
            col = layout.column(align=True)
            col.label(text="No Tessera cache found", icon='INFO')
            col.operator("object.tessera_build", text="Build Tessera", icon='MOD_SOLIDIFY')


classes = (
    TESSERA_PT_object,
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    register_refresh_timer()


def unregister():
    unregister_refresh_timer()
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
