"""Probe: scattered-asset Tessera culling + instance counts (GUI mode).

Usage:
    blender.exe <scatter blend> --python tessera_scatter_probe.py
"""
import math
import sys
import time

import bpy
import mathutils


def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return window, area, region, space, space.region_3d
    return None, None, None, None, None


def redraw_gui_frame():
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def clear_stats():
    """Remove stale stats persisted in the .blend so we only read fresh values."""
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        for key in list(ob.keys()):
            if key.startswith("tessera_stat_") or key.startswith("tessera_total_"):
                del ob[key]


def collect_stats():
    rows = []
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        drawn = int(ob.get("tessera_stat_drawn_instances", -1))
        clusters = int(ob.get("tessera_stat_clusters", -1))
        if drawn != -1 or clusters != -1:
            rows.append((ob.name, clusters, drawn))
    return rows


def dump_setup():
    print("\n=== TESSERA SETUP (emitters) ===")
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        keys = {}
        for key in ob.keys():
            if key.startswith("tessera_"):
                keys[key] = ob[key]
        if keys:
            print(f"  {ob.name:<28} {keys}")


def run_probe():
    if bpy.app.background:
        print("FATAL: GUI mode required")
        sys.exit(1)

    window, area, region, space, rv3d = find_view3d()
    if area is None:
        return 1.0

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    space.shading.type = 'SOLID'
    rv3d.view_perspective = 'PERSP'

    clear_stats()
    dump_setup()

    mesh_objects = [ob for ob in scene.objects if ob.type == 'MESH']

    # Enable Tessera on every mesh object exactly like gui_viewport_benchmark.py Phase 2.
    import tessera.ui
    for ob in mesh_objects:
        nc2 = tessera.ui.get_existing_nc2_path(ob)
        if nc2:
            ob["tessera_nc2"] = nc2
            ob["tessera_nc2_path"] = nc2
        ob["tessera_enabled"] = 1
        ob["tessera_visualize_clusters"] = 0
        ob["tessera_pixel_error"] = 1.0
        ob["tessera_culling_margin"] = 0.02
        if hasattr(ob, "tessera"):
            ob.tessera.enabled = True
        ob["tessera_show_stats"] = 1
    scene["tessera_navigating"] = 0
    bpy.context.view_layer.update()
    redraw_gui_frame()

    all_min = mathutils.Vector((1e9, 1e9, 1e9))
    all_max = mathutils.Vector((-1e9, -1e9, -1e9))
    for ob in mesh_objects:
        for b in ob.bound_box:
            w = ob.matrix_world @ mathutils.Vector(b)
            for i in range(3):
                all_min[i] = min(all_min[i], w[i])
                all_max[i] = max(all_max[i], w[i])
    center = 0.5 * (all_min + all_max)
    diag = (all_max - all_min).length

    # --- On-screen view ---
    rv3d.view_location = center
    rv3d.view_distance = max(diag * 0.75, 4.0)
    rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, 0.0), 'XYZ').to_quaternion()
    time.sleep(0.25)
    redraw_gui_frame()
    redraw_gui_frame()
    time.sleep(0.6)
    redraw_gui_frame()
    redraw_gui_frame()
    print("\n=== ON-SCREEN (viewport looking at field) ===")
    total_drawn = 0
    for name, clusters, drawn in collect_stats():
        if drawn >= 0 or clusters >= 0:
            print(f"  {name:<28} clusters={clusters:>6} drawn_instances={drawn:>6}")
            total_drawn += max(drawn, 0)
    print(f"  TOTAL drawn instances: {total_drawn:,}")

    # --- Off-screen view (pan far away) ---
    rv3d.view_location = center + mathutils.Vector((diag * 20.0, diag * 20.0, 0.0))
    rv3d.view_distance = 2.0
    time.sleep(0.25)
    redraw_gui_frame()
    redraw_gui_frame()
    time.sleep(0.6)
    redraw_gui_frame()
    redraw_gui_frame()
    print("\n=== OFF-SCREEN (viewport panned far away) ===")
    clear_stats()  # drop on-screen values so only fresh writes show up
    total_drawn_off = 0
    total_clusters_off = 0
    for name, clusters, drawn in collect_stats():
        if drawn >= 0 or clusters >= 0:
            print(f"  {name:<28} clusters={clusters:>6} drawn_instances={drawn:>6}")
            total_drawn_off += max(drawn, 0)
            total_clusters_off += max(clusters, 0)
    print(f"  TOTAL drawn instances: {total_drawn_off:,}  (expected 0)")
    print(f"  TOTAL reported clusters: {total_clusters_off:,}")

    bpy.ops.wm.quit_blender()
    sys.exit(0)
    return None


bpy.app.timers.register(run_probe, first_interval=0.5)
