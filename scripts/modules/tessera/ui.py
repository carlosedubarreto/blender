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


def is_tessera_object(ob):
    """Check if an object qualifies as a Tessera object."""
    if not ob or ob.type != 'MESH':
        return False
    if hasattr(ob, "tessera") and getattr(ob.tessera, "enabled", False):
        return True
    if bool(ob.get("tessera_enabled", False)):
        return True
    if ob.get("tessera_nc2") or ob.get("tessera_nc2_path"):
        return True
    if _nc2_path_cache.get(ob.name):
        return True
    from tessera.ui import get_existing_nc2_path
    if get_existing_nc2_path(ob):
        return True
    return False


def _get_prop(target, key, default=0.0):
    """Safely retrieve statistic property from original or evaluated datablock."""
    if target is None:
        return default
    val = target.get(key)
    if val is not None:
        return val
    try:
        if hasattr(target, "evaluated_get") and bpy.context and hasattr(bpy.context, "evaluated_depsgraph_get"):
            eval_target = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
            if eval_target and key in eval_target:
                return eval_target[key]
    except Exception:
        pass
    return default

def draw_tessera_hud():
    """Render on-screen viewport HUD overlay for active Tessera object or scene."""
    if bpy.app.background:
        return

    context = bpy.context
    scene = getattr(context, "scene", None)
    ob = getattr(context, "active_object", None)

    scene_hud = False
    if scene:
        sprops = getattr(scene, "tessera", None)
        scene_hud = bool(getattr(sprops, "show_fps_hud", False)) or bool(scene.get("tessera_show_fps_hud", 0))

    has_tessera_ob = is_tessera_object(ob)
    ob_hud = False
    if has_tessera_ob:
        oprops = getattr(ob, "tessera", None)
        ob_hud = bool(getattr(oprops, "show_fps_hud", False)) or bool(_get_prop(ob, "tessera_show_fps_hud", 0))

    if not (scene_hud or ob_hud):
        return

    fps = float(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_fps", _get_prop(scene, "tessera_stat_fps", 0.0)))
    ft_ms = float(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_frametime", _get_prop(scene, "tessera_stat_frametime", 0.0)))
    clusters = int(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_clusters", _get_prop(scene, "tessera_stat_clusters", 0)))
    tris = int(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_triangles", _get_prop(scene, "tessera_stat_triangles", 0)))

    if has_tessera_ob:
        from tessera.render import get_effective_margin
        eff_margin = float(_get_prop(ob, "tessera_stat_culling_margin", get_effective_margin(ob, context)))
        is_cam_active = (_get_prop(ob, "tessera_view_source", 0) == 1) or is_viewport_in_camera_view(context)
        mode_str = f"{ob.name} | Camera Locked" if is_cam_active else f"{ob.name} | Viewport Active"
        scale = float(_get_prop(ob, "tessera_stat_tex_scale", 1.0))
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
    else:
        eff_margin = float(_get_prop(scene, "tessera_culling_margin", 0.02))
        mode_str = "Scene Master"
        tex_str = "Auto"

    font_id = 0
    x_offset = 20
    region_h = context.region.height if context.region else 600

    vram_res = float(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_vram_resident_mb", _get_prop(scene, "tessera_stat_vram_resident_mb", 0.0)))
    budget_mb = float(getattr(scene, "tessera_vram_budget_mb", 1024)) if scene else 1024.0
    tot_inst = int(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_total_instances", _get_prop(scene, "tessera_stat_total_instances", 1)))
    act_inst = int(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_active_instances", _get_prop(scene, "tessera_stat_active_instances", 1)))

    def format_count(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return f"{n:,}"

    inst_str = format_count(tot_inst)
    active_inst_str = format_count(act_inst)

    text = (f"Tessera: {fps:.1f} FPS ({ft_ms:.2f} ms) | {clusters:,} clusters | {tris:,} tris | "
            f"Margin: {eff_margin:.2f}m ({mode_str}) | Tex: {tex_str} | "
            f"Mem: {vram_res:.0f}/{budget_mb:.0f}MB | Inst: {inst_str} ({active_inst_str} vis)")

    try:
        blf.size(font_id, 14)
        text_w, text_h = blf.dimensions(font_id, text)
        pad_x = 8
        pad_y = 5

        # Determine HUD placement from object or scene properties
        sprops = getattr(scene, "tessera", None) if scene else None
        oprops = getattr(ob, "tessera", None) if has_tessera_ob else None
        active_props = oprops if (oprops and getattr(oprops, "show_fps_hud", False)) else sprops
        if active_props is not None:
            pos = getattr(active_props, "hud_position", 'TOP')
            margin_y = int(getattr(active_props, "hud_margin_y", 60))
        elif scene:
            pos = str(scene.get("tessera_hud_position", "TOP"))
            margin_y = int(scene.get("tessera_hud_margin_y", 60))
        else:
            pos = 'TOP'
            margin_y = 60

        if pos == 'BOTTOM':
            y_offset = margin_y + pad_y
        else:
            y_offset = region_h - margin_y - int(text_h) - pad_y

        y_offset = max(pad_y + 2, min(region_h - int(text_h) - pad_y - 2, y_offset))

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
        blf.color(font_id, 0.2, 0.9, 0.4, 1.0)
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
    """Ensure HUD draw handler is registered only when at least one object or the scene has HUD enabled."""
    has_hud = False
    try:
        scene = bpy.context.scene if bpy.context else None
        if scene:
            sprops = getattr(scene, "tessera", None)
            if sprops and getattr(sprops, "show_fps_hud", False):
                has_hud = True
            elif bool(scene.get("tessera_show_fps_hud", 0)):
                has_hud = True

        if not has_hud:
            for ob in bpy.data.objects:
                props = getattr(ob, "tessera", None)
                if props and getattr(props, "show_fps_hud", False):
                    has_hud = True
                    break
                elif bool(ob.get("tessera_show_fps_hud", 0)):
                    has_hud = True
                    break
    except Exception:
        pass

    if has_hud:
        register_hud_handler()
    else:
        unregister_hud_handler()


def _tessera_sidebar_refresh_timer():
    """Continuous timer keeping sidebar telemetry and HUD overlay updated dynamically."""
    try:
        context = bpy.context
        scene = getattr(context, "scene", None)
        ob = getattr(context, "active_object", None)

        is_building = False
        try:
            is_building = any(bool(o.get("tessera_building", 0)) for o in bpy.data.objects)
        except Exception:
            pass

        sprops = getattr(scene, "tessera", None) if scene else None
        has_tessera_ob = is_tessera_object(ob)
        oprops = getattr(ob, "tessera", None) if has_tessera_ob else None

        scene_stats = bool(getattr(sprops, "show_stats", False)) or bool(scene.get("tessera_show_stats", 0) if scene else 0)
        scene_fps = bool(getattr(sprops, "show_fps_panel", False)) or bool(scene.get("tessera_show_fps_panel", 0) if scene else 0)
        scene_hud = bool(getattr(sprops, "show_fps_hud", False)) or bool(scene.get("tessera_show_fps_hud", 0) if scene else 0)

        ob_stats = bool(getattr(oprops, "show_stats", False)) or bool(ob.get("tessera_show_stats", 0) if ob else 0)
        ob_fps = bool(getattr(oprops, "show_fps_panel", False)) or bool(ob.get("tessera_show_fps_panel", 0) if ob else 0)
        ob_hud = bool(getattr(oprops, "show_fps_hud", False)) or bool(ob.get("tessera_show_fps_hud", 0) if ob else 0)

        needs_ui = is_building or scene_stats or scene_fps or ob_stats or ob_fps
        needs_render = scene_hud or ob_hud or scene_fps or ob_fps or scene_stats or ob_stats

        # If telemetry or builds are active, tag regions for smooth redraws
        if needs_ui or needs_render:
            wm = getattr(context, "window_manager", None)
            if wm:
                for window in wm.windows:
                    screen = getattr(window, "screen", None)
                    if screen:
                        for area in screen.areas:
                            if area.type == 'VIEW_3D':
                                for region in area.regions:
                                    if region.type == 'UI' and needs_ui:
                                        region.tag_redraw()
                                    if region.type == 'WINDOW' and needs_render:
                                        region.tag_redraw()
            return 0.1  # 10 FPS dynamic refresh rate
    except Exception:
        pass

    # Completely dormant when all telemetry is disabled
    return 2.0


def register_refresh_timer():
    """Ensure the continuous sidebar refresh timer is registered."""
    if not bpy.app.timers.is_registered(_tessera_sidebar_refresh_timer):
        bpy.app.timers.register(_tessera_sidebar_refresh_timer, persistent=True)


def unregister_refresh_timer():
    """Unregister the continuous sidebar refresh timer."""
    if bpy.app.timers.is_registered(_tessera_sidebar_refresh_timer):
        bpy.app.timers.unregister(_tessera_sidebar_refresh_timer)


_nc2_path_cache = {}
_nc2_size_cache = {}
_scene_stats_cache = {
    "total_meshes": 0,
    "available": 0,
    "enabled": 0,
    "synced": False,
}


def clear_nc2_path_cache():
    """Clear in-memory NC2 path and size cache."""
    _nc2_path_cache.clear()
    _nc2_size_cache.clear()
    _scene_stats_cache["synced"] = False


def sync_scene_nc2_status(scene=None):
    """Actively inspect all scene objects on disk and update status cache."""
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return 0, 0

    clear_nc2_path_cache()
    blend_dir = bpy.path.abspath("//")
    cwd = os.getcwd()

    total_meshes = 0
    tessera_available = 0
    tessera_enabled = 0

    for ob in scene.objects:
        if ob.type != 'MESH':
            continue
        total_meshes += 1

        nc2_path = None
        prop_path = ob.get("tessera_nc2", "") or ob.get("tessera_nc2_path", "")
        if prop_path:
            abs_p = bpy.path.abspath(prop_path)
            if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 88:
                nc2_path = abs_p

        if not nc2_path:
            mesh = getattr(ob, "data", None)
            mesh_name = mesh.name if mesh else ""
            candidates = [
                os.path.join(blend_dir, f"{ob.name}.nc2"),
                os.path.join(blend_dir, f"{mesh_name}.nc2") if mesh_name else "",
                os.path.join(blend_dir, f"{ob.name.replace('.', '_')}.nc2"),
                os.path.join(blend_dir, f"{mesh_name.replace('.', '_')}.nc2") if mesh_name else "",
                os.path.join(cwd, f"{ob.name}.nc2"),
                os.path.join(cwd, f"{mesh_name}.nc2") if mesh_name else "",
            ]
            for c in candidates:
                if not c:
                    continue
                abs_p = bpy.path.abspath(c)
                if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 88:
                    nc2_path = abs_p
                    break

        if nc2_path:
            _nc2_path_cache[ob.name] = nc2_path
            try:
                _nc2_size_cache[ob.name] = os.path.getsize(nc2_path) / (1024 * 1024)
            except Exception:
                _nc2_size_cache[ob.name] = 0.0
            tessera_available += 1
        else:
            _nc2_path_cache[ob.name] = ""
            _nc2_size_cache[ob.name] = 0.0

        is_en = False
        if hasattr(ob, "tessera") and getattr(ob.tessera, "enabled", False):
            is_en = True
        elif bool(ob.get("tessera_enabled", False)):
            is_en = True
        if is_en:
            tessera_enabled += 1

    _scene_stats_cache["total_meshes"] = total_meshes
    _scene_stats_cache["available"] = tessera_available
    _scene_stats_cache["enabled"] = tessera_enabled
    _scene_stats_cache["synced"] = True

    wm = getattr(bpy.context, "window_manager", None)
    if wm:
        for window in wm.windows:
            screen = getattr(window, "screen", None)
            if screen:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()

    return tessera_available, total_meshes


def get_existing_nc2_path(ob):
    """Detect whether a valid compiled Tessera .nc2 binary exists on disk."""
    if ob is None:
        return None

    # Check object properties first (fastest, in-memory)
    prop_path = ob.get("tessera_nc2", "") or ob.get("tessera_nc2_path", "")
    if prop_path:
        abs_p = bpy.path.abspath(prop_path)
        scene = getattr(bpy.context, "scene", None)
        auto_checks = getattr(getattr(scene, "tessera", None), "auto_file_checks", False) if scene else False
        if not auto_checks:
            _nc2_path_cache[ob.name] = abs_p
            return abs_p
        if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 88:
            _nc2_path_cache[ob.name] = abs_p
            return abs_p

    cached = _nc2_path_cache.get(ob.name)
    if cached is not None:
        return cached if cached else None

    scene = getattr(bpy.context, "scene", None)
    auto_checks = getattr(getattr(scene, "tessera", None), "auto_file_checks", False) if scene else False
    if not auto_checks:
        _nc2_path_cache[ob.name] = ""
        return None

    blend_dir = bpy.path.abspath("//")
    mesh = getattr(ob, "data", None)
    mesh_name = mesh.name if mesh else ""

    candidates = [
        os.path.join(blend_dir, f"{ob.name}.nc2"),
        os.path.join(blend_dir, f"{mesh_name}.nc2") if mesh_name else "",
        os.path.join(blend_dir, f"{ob.name.replace('.', '_')}.nc2"),
        os.path.join(blend_dir, f"{mesh_name.replace('.', '_')}.nc2") if mesh_name else "",
        os.path.join(os.getcwd(), f"{ob.name}.nc2"),
        os.path.join(os.getcwd(), f"{mesh_name}.nc2") if mesh_name else "",
    ]

    for c in candidates:
        if not c:
            continue
        abs_p = bpy.path.abspath(c)
        if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 88:
            _nc2_path_cache[ob.name] = abs_p
            return abs_p

    _nc2_path_cache[ob.name] = ""
    return None


class TESSERA_PT_telemetry(bpy.types.Panel):
    """Realtime telemetry and performance telemetry panel, always visible."""

    bl_label = "Realtime Telemetry"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        sprops = getattr(scene, "tessera", None)
        ob = getattr(context, "active_object", None)
        has_tessera_ob = is_tessera_object(ob)
        oprops = getattr(ob, "tessera", None) if has_tessera_ob else None

        # Telemetry Toggles (Always visible, persistent)
        t_row = layout.row(align=True)
        t_row.prop(sprops, "show_stats", text="Live Stats", icon='INFO')
        t_row.prop(sprops, "log_console", text="Console Log", icon='CONSOLE')

        fps_row = layout.row(align=True)
        fps_row.prop(sprops, "show_fps_panel", text="Panel FPS", icon='TIME')
        fps_row.prop(sprops, "show_fps_hud", text="Viewport HUD", icon='RESTRICT_VIEW_OFF')

        show_hud = getattr(sprops, "show_fps_hud", False) or (oprops and getattr(oprops, "show_fps_hud", False))
        if show_hud:
            hud_box = layout.box()
            hud_box.label(text="HUD Settings:", icon='PREFERENCES')
            h_row = hud_box.row(align=True)
            target_props = oprops if (has_tessera_ob and getattr(oprops, "show_fps_hud", False)) else sprops
            h_row.prop(target_props, "hud_position", expand=True)
            h_row.prop(target_props, "hud_margin_y", text="Distance (px)")

        show_fps = getattr(sprops, "show_fps_panel", False) or (oprops and getattr(oprops, "show_fps_panel", False))
        show_stats = getattr(sprops, "show_stats", False) or (oprops and getattr(oprops, "show_stats", False))

        if show_fps:
            box = layout.box()
            fps = float(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_fps", _get_prop(scene, "tessera_stat_fps", 0.0)))
            ms = float(_get_prop(ob if has_tessera_ob else scene, "tessera_stat_frametime", _get_prop(scene, "tessera_stat_frametime", 0.0)))
            col = box.column(align=True)
            col.label(text=f"Performance: {fps:.1f} FPS ({ms:.2f} ms)", icon='PREFERENCES')

        if show_stats:
            box = layout.box()
            col = box.column(align=True)

            tot_clusters = int(_get_prop(scene, "tessera_total_clusters", 0))
            act_clusters = int(_get_prop(scene, "tessera_stat_clusters", 0))
            tot_tris = int(_get_prop(scene, "tessera_total_triangles", 0))
            act_tris = int(_get_prop(scene, "tessera_stat_triangles", 0))

            vram_res = float(_get_prop(scene, "tessera_stat_vram_resident_mb", 0.0))
            budget_mb = float(getattr(scene, "tessera_vram_budget_mb", 1024))
            tot_inst = int(_get_prop(scene, "tessera_stat_total_instances", 0))
            act_inst = int(_get_prop(scene, "tessera_stat_active_instances", 0))

            col.label(text="Scene Live Stats:", icon='SCENE_DATA')
            if tot_inst > 0:
                col.label(text=f"Instances: {act_inst:,} active / {tot_inst:,} total", icon='OUTLINER_OB_GROUP_INSTANCE')
            if act_clusters > 0 or tot_clusters > 0:
                col.label(text=f"Clusters: {act_clusters:,} active / {tot_clusters:,} total", icon='MOD_SOLIDIFY')
            if act_tris > 0 or tot_tris > 0:
                col.label(text=f"Triangles: {act_tris:,} active / {tot_tris:,} total", icon='MESH_DATA')
            if budget_mb > 0:
                col.label(text=f"VRAM: {vram_res:.1f} MB / {budget_mb:.0f} MB", icon='DISK_DRIVE')

            if has_tessera_ob:
                col.separator()
                col.label(text=f"Active Object: {ob.name}", icon='OBJECT_DATA')
                lod_min = int(_get_prop(ob, 'tessera_stat_lod_min', 0))
                lod_max = int(_get_prop(ob, 'tessera_stat_lod_max', 0))
                col.label(text=f"Active LODs: {lod_min} – {lod_max}", icon='ZOOM_IN')

                active_clusters = int(_get_prop(ob, 'tessera_stat_clusters', 0))
                total_clusters = int(_get_prop(ob, 'tessera_total_clusters', 0))
                col.label(text=f"Clusters: {active_clusters:,} / {total_clusters:,}", icon='MOD_SOLIDIFY')

                active_tris = int(_get_prop(ob, 'tessera_stat_triangles', 0))
                total_tris = int(_get_prop(ob, 'tessera_total_triangles', 0))
                col.label(text=f"Triangles: {active_tris:,} / {total_tris:,}", icon='MESH_DATA')

                eff_margin = float(_get_prop(ob, "tessera_stat_culling_margin", 0.0))
                col.label(text=f"Culling Margin: {eff_margin:.2f}m", icon='VIEW_CAMERA')
            else:
                col.label(text="(Select a Tessera mesh for per-object LOD details)", icon='INFO')


class TESSERA_PT_object(bpy.types.Panel):
    """Tessera options for the active mesh object."""

    bl_label = "Tessera Object Telemetry & Settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        if ob is None or ob.type != 'MESH':
            box = layout.box()
            box.label(text="Select a mesh object for individual telemetry and manual overrides.", icon='INFO')
            return

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

            cached_size = _nc2_size_cache.get(ob.name)
            if cached_size is not None and cached_size > 0:
                size_str = f"{cached_size:.1f} MB"
            else:
                try:
                    s_mb = os.path.getsize(nc2_path) / (1024 * 1024)
                    _nc2_size_cache[ob.name] = s_mb
                    size_str = f"{s_mb:.1f} MB"
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
            if ob.get("tessera_is_proxy", 0):
                col.operator("tessera.restore_mesh", text="Restore Original Mesh", icon='MESH_DATA')
            else:
                col.operator("tessera.convert_to_proxy", text="Convert Base Mesh to Lightweight Proxy (Keep .nc2)", icon='MESH_CUBE')
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
                prop_col.prop(tessera_props, "lod_distance", text="LOD Distance")
                prop_col.prop(tessera_props, "lod_intensity", text="LOD Intensity")
                prop_col.prop(tessera_props, "scatter_pixel_error", text="Scatter Pixel Error")
                prop_col.prop(tessera_props, "min_screen_radius", text="Min Screen Radius")
                prop_col.prop(tessera_props, "max_draw_distance", text="Max Draw Distance")
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

                if tessera_props.show_fps_hud:
                    h_row = t_box.row(align=True)
                    h_row.prop(tessera_props, "hud_position", expand=True)
                    h_row.prop(tessera_props, "hud_margin_y", text="Distance (px)")

                if tessera_props.show_fps_panel:
                    fps_col = t_box.column(align=True)
                    fps = float(_get_prop(ob, "tessera_stat_fps", 0.0))
                    ms = float(_get_prop(ob, "tessera_stat_frametime", 0.0))
                    fps_col.label(text=f"Performance: {fps:.1f} FPS ({ms:.2f} ms)", icon='PREFERENCES')

                if tessera_props.show_stats:
                    t_col = t_box.column(align=True)
                    lod_min = _get_prop(ob, 'tessera_stat_lod_min', 0)
                    lod_max = _get_prop(ob, 'tessera_stat_lod_max', 0)
                    t_col.label(text=f"Active LODs: {lod_min} – {lod_max}", icon='ZOOM_IN')

                    active_clusters = _get_prop(ob, 'tessera_stat_clusters', 0)
                    total_clusters = _get_prop(ob, 'tessera_total_clusters', 0)
                    t_col.label(text=f"Visible Clusters: {active_clusters:,} / {total_clusters:,}", icon='MOD_SOLIDIFY')

                    active_tris = _get_prop(ob, 'tessera_stat_triangles', 0)
                    total_tris = _get_prop(ob, 'tessera_total_triangles', 0)
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


class TESSERA_PT_batch_controls(bpy.types.Panel):
    """Universal Scene Batch Controls."""

    bl_label = "Batch Controls"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_order = 0

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        t_scene = getattr(scene, "tessera", None)
        if not t_scene:
            return

        # 1. Scene Mesh Statistics Overview
        stat_box = layout.box()
        h_row = stat_box.row(align=True)
        h_row.label(text="Scene Mesh Overview", icon='MESH_DATA')
        h_row.operator("tessera.sync_status", text="Sync Cache Status", icon='FILE_REFRESH')

        auto_checks = getattr(t_scene, "auto_file_checks", False)

        if not _scene_stats_cache.get("synced", False) and not auto_checks:
            total_meshes = 0
            tessera_available = 0
            tessera_enabled = 0
            for ob in scene.objects:
                if ob.type != 'MESH':
                    continue
                total_meshes += 1
                if ob.get("tessera_nc2") or ob.get("tessera_nc2_path") or _nc2_path_cache.get(ob.name):
                    tessera_available += 1
                is_en = False
                if hasattr(ob, "tessera") and getattr(ob.tessera, "enabled", False):
                    is_en = True
                elif bool(ob.get("tessera_enabled", False)):
                    is_en = True
                if is_en:
                    tessera_enabled += 1
        elif auto_checks:
            total_meshes = 0
            tessera_available = 0
            tessera_enabled = 0
            for ob in scene.objects:
                if ob.type != 'MESH':
                    continue
                total_meshes += 1
                if get_existing_nc2_path(ob) is not None:
                    tessera_available += 1
                is_en = False
                if hasattr(ob, "tessera") and getattr(ob.tessera, "enabled", False):
                    is_en = True
                elif bool(ob.get("tessera_enabled", False)):
                    is_en = True
                if is_en:
                    tessera_enabled += 1
        else:
            total_meshes = _scene_stats_cache.get("total_meshes", 0)
            tessera_available = _scene_stats_cache.get("available", 0)
            tessera_enabled = _scene_stats_cache.get("enabled", 0)

        stat_col = stat_box.column(align=True)
        stat_col.label(text=f"Total Mesh Objects: {total_meshes}", icon='OBJECT_DATAMODE')
        stat_col.label(text=f"Tessera Available (.nc2): {tessera_available} / {total_meshes}", icon='FILE_CACHE')
        stat_col.label(
            text=f"Tessera Enabled: {tessera_enabled} / {total_meshes}",
            icon='CHECKBOX_HLT' if tessera_enabled > 0 else 'CHECKBOX_DEHLT'
        )

        sync_row = stat_box.row(align=True)
        sync_row.prop(t_scene, "auto_file_checks", text="Auto Check Files (Disk I/O)")

        # 2. Enable / Disable All Buttons (Replacing "Batch Tessera Enabled")
        row_btn = layout.row(align=True)
        row_btn.scale_y = 1.3
        row_btn.operator("tessera.batch_enable_all", text="Enable All Tessera", icon='CHECKBOX_HLT')
        row_btn.operator("tessera.batch_disable_all", text="Disable All Tessera", icon='CHECKBOX_DEHLT')

        # 3. Batch Scope Box (for Selection-specific actions)
        box = layout.box()
        box.label(text="Batch Scope", icon='VIEWZOOM')
        box.prop(t_scene, "batch_scope", expand=True)

        if t_scene.batch_scope == 'SELECTION':
            r = box.row(align=True)
            r.operator("tessera.batch_enable_selected", text="Enable Selected", icon='RESTRICT_SELECT_OFF')
            r.operator("tessera.batch_disable_selected", text="Disable Selected", icon='RESTRICT_SELECT_ON')

        # 4. Batch Properties (Tuning Sliders)
        b_col = layout.column(align=True)
        b_col.prop(t_scene, "batch_pixel_error", text="Batch Pixel Error", slider=True)
        b_col.prop(t_scene, "batch_culling_margin", text="Batch Camera Margin")
        b_col.prop(t_scene, "batch_lod_distance", text="Batch LOD Distance")
        b_col.prop(t_scene, "batch_lod_intensity", text="Batch LOD Intensity")
        b_col.prop(t_scene, "batch_scatter_pixel_error", text="Batch Scatter Pixel Error")
        b_col.prop(t_scene, "batch_min_screen_radius", text="Batch Min Screen Radius")
        b_col.prop(t_scene, "batch_max_draw_distance", text="Batch Max Draw Distance")
        b_col.prop(t_scene, "batch_texture_lod_mode", text="Batch Texture LOD")
        b_col.prop(t_scene, "batch_use_camera_view", text="Batch Camera Locked LOD", icon='CAMERA_DATA')


class TESSERA_PT_tree_view(bpy.types.Panel):
    """Collapsible Collection & Object Tree View."""

    bl_label = "Tessera Scene Tree"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        t_scene = getattr(scene, "tessera", None)

        row = layout.row(align=True)
        if t_scene:
            row.prop(t_scene, "tree_display_mode", expand=True)
            mode = t_scene.tree_display_mode
        else:
            mode = 'COLLECTIONS'
        row.operator("tessera.sync_status", text="", icon='FILE_REFRESH')

        if mode == 'COLLECTIONS':
            collections = list(scene.collection.children)
            if not collections:
                collections = [scene.collection]
            elif any(o.type == 'MESH' for o in scene.collection.objects):
                if scene.collection not in collections:
                    collections.insert(0, scene.collection)

            for col in collections:
                mesh_obs = [o for o in col.all_objects if o.type == 'MESH']
                is_expanded = bool(col.get("tessera_tree_expanded", True))

                # Aggregate Tessera state across member meshes (tri-state):
                # all enabled / none enabled / mixed. The RNA PropertyGroup is
                # authoritative when present; the raw IDProperty is the
                # fallback read by the C++ draw code.
                enabled_count = 0
                for o in mesh_obs:
                    if hasattr(o, "tessera"):
                        if bool(o.tessera.enabled):
                            enabled_count += 1
                    elif int(o.get("tessera_enabled", 0)) != 0:
                        enabled_count += 1
                if mesh_obs and enabled_count == len(mesh_obs):
                    state_icon = 'CHECKBOX_HLT'
                elif enabled_count == 0:
                    state_icon = 'CHECKBOX_DEHLT'
                else:
                    state_icon = 'QUESTION'  # mixed state

                c_box = layout.box()
                header_row = c_box.row(align=True)
                arrow_icon = 'TRIA_DOWN' if is_expanded else 'TRIA_RIGHT'
                op_toggle = header_row.operator("tessera.toggle_collection", text="", icon=arrow_icon, emboss=False)
                op_toggle.collection_name = col.name

                header_row.label(text=f"{col.name} ({len(mesh_obs)})", icon='OUTLINER_COLLECTION')
                op_build_col = header_row.operator("tessera.build_collection", text="", icon='MOD_BUILD')
                op_build_col.collection_name = col.name

                # Batch enable / disable switches for this collection.
                op_enable_col = header_row.operator("tessera.collection_set_enabled", text="", icon='CHECKBOX_HLT')
                op_enable_col.collection_name = col.name
                op_enable_col.enabled = True
                op_enable_col.recursive = True
                op_disable_col = header_row.operator("tessera.collection_set_enabled", text="", icon='CHECKBOX_DEHLT')
                op_disable_col.collection_name = col.name
                op_disable_col.enabled = False
                op_disable_col.recursive = True
                header_row.label(text="", icon=state_icon)

                if is_expanded and mesh_obs:
                    item_col = c_box.column(align=True)
                    for ob in mesh_obs:
                        i_row = item_col.row(align=True)
                        has_nc2 = get_existing_nc2_path(ob) is not None
                        icon_status = 'CHECKMARK' if has_nc2 else 'CANCEL'
                        i_row.label(text="", icon=icon_status)
                        i_row.label(text=ob.name, icon='MESH_DATA')

                        if hasattr(ob, "tessera"):
                            i_row.prop(ob.tessera, "enabled", text="")
                        else:
                            i_row.prop(ob, '["tessera_enabled"]', text="")

                        op_build_ob = i_row.operator("tessera.build_single_object", text="", icon='FILE_REFRESH')
                        op_build_ob.object_name = ob.name

        else:  # 'OBJECTS' (Flat Objects)
            mesh_obs = [o for o in scene.objects if o.type == 'MESH']
            if not mesh_obs:
                layout.label(text="No mesh objects in scene", icon='INFO')
                return

            box = layout.box()
            item_col = box.column(align=True)
            for ob in mesh_obs:
                i_row = item_col.row(align=True)
                has_nc2 = get_existing_nc2_path(ob) is not None
                icon_status = 'CHECKMARK' if has_nc2 else 'CANCEL'
                i_row.label(text="", icon=icon_status)
                i_row.label(text=ob.name, icon='MESH_DATA')

                if hasattr(ob, "tessera"):
                    i_row.prop(ob.tessera, "enabled", text="")
                else:
                    i_row.prop(ob, '["tessera_enabled"]', text="")

                op_build_ob = i_row.operator("tessera.build_single_object", text="", icon='FILE_REFRESH')
                op_build_ob.object_name = ob.name


class TESSERA_PT_scene_manager(bpy.types.Panel):
    """Scene Batch & Memory Manager."""

    bl_label = "Scene Batch & Memory Manager"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_order = 4

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Batch Controls Box
        box = layout.box()
        box.label(text="Batch Controls", icon='GROUP')
        col = box.column(align=True)
        r_all = col.row(align=True)
        r_all.operator("tessera.batch_enable_all", text="Enable All", icon='CHECKBOX_HLT')
        r_all.operator("tessera.batch_disable_all", text="Disable All", icon='CHECKBOX_DEHLT')
        row = col.row(align=True)
        row.operator("tessera.batch_enable_selected", text="Enable Selected", icon='RESTRICT_SELECT_OFF')
        row.operator("tessera.batch_disable_selected", text="Disable Selected", icon='RESTRICT_SELECT_ON')

        t_scene = getattr(scene, "tessera", None)
        if t_scene:
            box.prop(t_scene, "auto_file_checks", text="Auto Check Files")
        box.prop(scene, "tessera_auto_build_missing", text="Auto-Build Missing Caches")
        row_sync = box.row(align=True)
        row_sync.operator("tessera.sync_status", text="Sync Cache Status", icon='FILE_REFRESH')
        row_sync.operator("tessera.batch_build_all", text="Batch Build All Missing .nc2", icon='FILE_CACHE')
        prow = box.row(align=True)
        prow.operator("tessera.convert_to_proxy", text="Convert to Proxy (Keep .nc2)", icon='MESH_CUBE')
        prow.operator("tessera.restore_mesh", text="Restore Original Mesh", icon='FILE_REFRESH')

        # Memory Budget & Streaming Box
        sbox = layout.box()
        sbox.label(text="Streaming & Memory Budget", icon='MEMORY')
        sbox.prop(scene, "tessera_vram_budget_mb", text="VRAM Budget (MB)", slider=True)

        ob = context.active_object
        vram_res = float(ob.get("tessera_stat_vram_resident_mb", scene.get("tessera_stat_vram_resident_mb", 0.0)) if ob else scene.get("tessera_stat_vram_resident_mb", 0.0))
        budget_mb = float(getattr(scene, "tessera_vram_budget_mb", 1024))
        pct = (vram_res / max(budget_mb, 1.0)) * 100.0

        disk_mbs = float(ob.get("tessera_stat_disk_read_mbs", scene.get("tessera_stat_disk_read_mbs", 0.0)) if ob else scene.get("tessera_stat_disk_read_mbs", 0.0))
        hit_rate = float(ob.get("tessera_stat_cache_hit_ratio", scene.get("tessera_stat_cache_hit_ratio", 100.0)) if ob else scene.get("tessera_stat_cache_hit_ratio", 100.0))

        tot_inst = int(ob.get("tessera_stat_total_instances", scene.get("tessera_stat_total_instances", 0)) if ob else scene.get("tessera_stat_total_instances", 0))
        act_inst = int(ob.get("tessera_stat_active_instances", scene.get("tessera_stat_active_instances", 0)) if ob else scene.get("tessera_stat_active_instances", 0))

        tcol = sbox.column(align=True)
        tcol.label(text=f"VRAM Resident: {vram_res:.1f} MB / {budget_mb:.0f} MB ({pct:.1f}%)", icon='DISK_DRIVE')
        tcol.label(text=f"Disk Pre-fetch: {disk_mbs:.1f} MB/s | Hit Rate: {hit_rate:.1f}%", icon='SORTTIME')
        tcol.label(text=f"Instances: {tot_inst:,} (Active: {act_inst:,})", icon='DUPLICATE')


classes = (
    TESSERA_PT_batch_controls,
    TESSERA_PT_telemetry,
    TESSERA_PT_tree_view,
    TESSERA_PT_object,
    TESSERA_PT_scene_manager,
)


@bpy.app.handlers.persistent
def _on_load_post(dummy):
    clear_nc2_path_cache()
    register_refresh_timer()
    update_hud_state()
    # Strictly respect the saved state on file load!
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            if "tessera_enabled" in ob:
                saved_state = bool(ob["tessera_enabled"])
                if hasattr(ob, "tessera"):
                    try:
                        ob.tessera.enabled = saved_state
                    except Exception:
                        pass
    scene = getattr(bpy.context, "scene", None)
    if scene:
        if "tessera_batch_enabled" in scene:
            saved_batch = bool(scene["tessera_batch_enabled"])
            if hasattr(scene, "tessera"):
                try:
                    scene.tessera.batch_enabled = saved_batch
                except Exception:
                    pass


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    register_refresh_timer()
    update_hud_state()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def unregister():
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    unregister_hud_handler()
    unregister_refresh_timer()
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

