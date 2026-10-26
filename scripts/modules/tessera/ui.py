"""Tessera user interface.

Adds a *Tessera* panel to the 3D viewport sidebar (N-panel) exposing the
available Tessera tools: building cluster data from the active mesh, loading
previously saved ``.nc2`` files, and toggling cluster visualization.

The panel is automatically registered together with the rest of the
:mod:`tessera` module by :func:`tessera.register`.
"""

import os
import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_hud_handler = None


def is_viewport_in_camera_view(context):
    """Check whether the active 3D Viewport is currently looking through the camera."""
    space = getattr(context, "space_data", None)
    if space and getattr(space, "type", None) == 'VIEW_3D':
        rv3d = getattr(space, "region_3d", None)
        if rv3d and getattr(rv3d, "view_perspective", None) == 'CAMERA':
            return True
    return False


def draw_tessera_hud():
    """Render on-screen viewport HUD overlay for active Tessera object."""
    if bpy.app.background:
        return

    context = bpy.context
    ob = getattr(context, "active_object", None)
    if not ob or not ob.get("tessera_enabled", False):
        return

    props = getattr(ob, "tessera", None)
    if not props or not getattr(props, "show_fps_hud", False):
        return

    fps = float(ob.get("tessera_stat_fps", 0.0))
    ft_ms = float(ob.get("tessera_stat_frametime", 0.0))
    clusters = int(ob.get("tessera_stat_clusters", 0))
    tris = int(ob.get("tessera_stat_triangles", 0))

    from tessera.render import get_effective_margin
    eff_margin = float(ob.get("tessera_stat_culling_margin", get_effective_margin(ob, context)))
    is_cam_active = (ob.get("tessera_view_source", 0) == 1) or is_viewport_in_camera_view(context)
    mode_str = "Camera Locked" if is_cam_active else "Viewport Active"

    # Position in top-left margin (offset from header/gizmos)
    font_id = 0
    x_offset = 20
    region_h = context.region.height if context.region else 600
    y_offset = region_h - 60

    scale = float(ob.get("tessera_stat_tex_scale", 1.0))
    scale_pct = int(round(scale * 100))
    if scale >= 0.99:
        tex_str = "100% (Full)"
    elif scale >= 0.49:
        tex_str = "50% (1/2)"
    elif scale >= 0.24:
        tex_str = "25% (1/4)"
    elif scale >= 0.12:
        tex_str = "12.5% (1/8)"
    else:
        tex_str = f"{scale_pct}%"

    text = f"Tessera: {fps:.1f} FPS ({ft_ms:.2f} ms) | {clusters:,} clusters | {tris:,} tris | Margin: {eff_margin:.2f}m ({mode_str}) | Tex: {tex_str}"

    try:
        # Optional subtle background pill/box
        blf.size(font_id, 14)
        text_w, text_h = blf.dimensions(font_id, text)
        pad_x = 8
        pad_y = 5
        x0 = float(x_offset - pad_x)
        x1 = float(x_offset + text_w + pad_x)
        y0 = float(y_offset - pad_y)
        y1 = float(y_offset + text_h + pad_y)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(
            shader, 'TRI_FAN',
            {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]}
        )
        gpu.state.blend_set('ALPHA')
        shader.bind()
        shader.uniform_float("color", (0.05, 0.05, 0.08, 0.75))
        batch.draw(shader)
        gpu.state.blend_set('NONE')

        blf.size(font_id, 14)
        blf.color(font_id, 0.2, 0.9, 0.4, 1.0)  # Green accent
        blf.position(font_id, x_offset, y_offset, 0)
        blf.draw(font_id, text)
    except Exception:
        pass


def register_hud_handler():
    """Register the viewport HUD draw handler."""
    global _hud_handler
    if _hud_handler is None:
        _hud_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_tessera_hud, (), 'WINDOW', 'POST_PIXEL'
        )


def unregister_hud_handler():
    """Unregister the viewport HUD draw handler."""
    global _hud_handler
    if _hud_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_hud_handler, 'WINDOW')
        except Exception:
            pass
        _hud_handler = None


def update_hud_state():
    """Ensure HUD draw handler is registered only when at least one object has HUD enabled."""
    has_hud = False
    try:
        for ob in bpy.data.objects:
            props = getattr(ob, "tessera", None)
            if props and getattr(props, "show_fps_hud", False):
                has_hud = True
                break
    except Exception:
        pass

    if has_hud:
        register_hud_handler()
    else:
        unregister_hud_handler()


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
    needs_hud = False
    if ob and bool(ob.get("tessera_enabled", False)):
        props = getattr(ob, "tessera", None)
        if props:
            if getattr(props, "show_stats", False) or getattr(props, "show_fps_panel", False):
                needs_stats = True
            if getattr(props, "show_fps_hud", False):
                needs_hud = True
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
                                elif region.type == 'WINDOW' and needs_hud:
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
                # Camera View toggle button with dynamic alert state
                in_cam_perspective = is_viewport_in_camera_view(context)
                is_cam_lod_active = (ob.get("tessera_view_source", 0) == 1) or in_cam_perspective

                prop_col = layout.column(align=True)
                prop_col.use_property_split = True
                prop_col.use_property_decorate = False
                prop_col.prop(tessera_props, "pixel_error", text="Pixel Error")
                prop_col.prop(tessera_props, "culling_margin", text="Camera Margin")
                prop_col.prop(tessera_props, "texture_lod_mode", text="Texture LOD")
                prop_col.prop(tessera_props, "visualize_clusters")

                cam_row = layout.row(align=True)
                if in_cam_perspective and not (ob.get("tessera_view_source", 0) == 1):
                    cam_row.alert = True
                    cam_row.prop(
                        tessera_props,
                        "use_camera_view",
                        text="Enable Camera LOD (Numpad 0 Active)",
                        icon='CAMERA_DATA',
                        toggle=True
                    )
                else:
                    cam_row.alert = False
                    is_view_src_cam = (ob.get("tessera_view_source", 0) == 1)
                    btn_text = "Camera LOD: Active" if is_view_src_cam else "Camera LOD: Viewport"
                    btn_icon = 'CHECKBOX_HLT' if is_view_src_cam else 'CAMERA_DATA'
                    cam_row.prop(
                        tessera_props,
                        "use_camera_view",
                        text=btn_text,
                        icon=btn_icon,
                        toggle=True
                    )

                # Realtime Telemetry Sub-Panel
                t_box = layout.box()
                t_box.label(text="Realtime Telemetry", icon='VIEW_CAMERA')
                t_row = t_box.row(align=True)
                t_row.prop(tessera_props, "show_stats", text="Live Stats", icon='INFO')
                t_row.prop(tessera_props, "log_console", text="Console Log", icon='CONSOLE')

                fps_row = t_box.row(align=True)
                fps_row.prop(tessera_props, "show_fps_panel", text="Panel FPS", icon='TIME')
                fps_row.prop(tessera_props, "show_fps_hud", text="Viewport HUD", icon='RESTRICT_VIEW_OFF')

                if tessera_props.show_fps_panel:
                    fps_col = t_box.column(align=True)
                    fps = float(ob.get("tessera_stat_fps", 0.0))
                    ms = float(ob.get("tessera_stat_frametime", 0.0))
                    fps_col.label(text=f"Performance: {fps:.1f} FPS ({ms:.2f} ms)", icon='PREFERENCES')

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

                    scale = float(ob.get("tessera_stat_tex_scale", 1.0))
                    rx = int(ob.get("tessera_stat_tex_res_x", 0))
                    ry = int(ob.get("tessera_stat_tex_res_y", 0))
                    scale_pct = int(round(scale * 100))
                    res_str = f" - {rx}x{ry}" if (rx > 0 and ry > 0) else ""
                    if scale >= 0.99:
                        status_str = f"100% (Full Res{res_str})"
                    elif scale >= 0.49:
                        status_str = f"50% (Half Res{res_str})"
                    elif scale >= 0.24:
                        status_str = f"25% (Quarter Res{res_str})"
                    elif scale >= 0.12:
                        status_str = f"12.5% (Eighth Res{res_str})"
                    else:
                        status_str = f"{scale_pct}% ({rx}x{ry})"
                    t_col.label(text=f"Texture Scale: {status_str}", icon='IMAGE_DATA')

                    dist = ob.get('tessera_stat_distance', 0.0)
                    err = ob.get('tessera_stat_pixel_error', ob.get('tessera_stat_error', 1.0))
                    from tessera.render import get_effective_margin
                    eff_margin = float(ob.get('tessera_stat_culling_margin', get_effective_margin(ob, context)))
                    mode_str = "Camera Locked" if is_cam_lod_active else "Viewport Active"
                    t_col.label(text=f"Camera Distance: {dist:.2f} m", icon='CAMERA_DATA')
                    t_col.label(text=f"Projected Error: {err:.1f} px", icon='DRIVER_DISTANCE')
                    t_col.label(text=f"Active Margin: {eff_margin:.2f} m ({mode_str})", icon='FULLSCREEN_ENTER')

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
    register_hud_handler()


def unregister():
    unregister_hud_handler()
    unregister_refresh_timer()
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
