"""Sweep Benchmark for Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend.
Tests various settings to reach 20-30+ FPS.
"""

import ctypes
import math
import os
import sys
import time
import bpy
import mathutils

def get_gl_finish():
    try:
        opengl32 = ctypes.windll.opengl32
        opengl32.glFinish.argtypes = []
        opengl32.glFinish.restype = None
        return opengl32.glFinish
    except Exception:
        return lambda: None

gl_finish = get_gl_finish()

def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                region_win = None
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region_win = region
                        break
                return window, area, region_win, space, space.region_3d
    return None, None, None, None, None

def run_sweep():
    if bpy.app.background:
        print("FATAL: Benchmark must be run in interactive GUI mode (without -b)!")
        sys.exit(1)

    window, area, region, space, rv3d = find_view3d()
    if area is None or rv3d is None:
        return 0.1

    print("\n" + "=" * 80)
    print("LIVE INTERACTIVE GUI VIEWPORT BENCHMARK - PARAMETER SWEEP")
    print(f"Binary : {bpy.app.binary_path}")
    print(f"File   : {bpy.data.filepath}")
    print("=" * 80)

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    space.shading.type = 'SOLID'
    rv3d.view_perspective = 'PERSP'

    mesh_objects = [ob for ob in scene.objects if ob.type == 'MESH']
    
    # Compute scene center and orbit distance
    all_min = mathutils.Vector((1e9, 1e9, 1e9))
    all_max = mathutils.Vector((-1e9, -1e9, -1e9))
    for ob in mesh_objects:
        if ob.hide_viewport:
            continue
        for b in ob.bound_box:
            w = ob.matrix_world @ mathutils.Vector(b)
            for i in range(3):
                all_min[i] = min(all_min[i], w[i])
                all_max[i] = max(all_max[i], w[i])
    center = 0.5 * (all_min + all_max)
    diag = (all_max - all_min).length
    orbit_dist = max(diag * 0.75, 4.0)

    rv3d.view_location = center
    rv3d.view_distance = orbit_dist

    override = {
        "window": window,
        "screen": window.screen,
        "area": area,
        "region": region,
        "space_data": space,
        "region_data": rv3d,
    }

    def redraw_gui_frame():
        with bpy.context.temp_override(**override):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def run_orbit(label, frames=30):
        # Warmup 3 frames
        for i in range(3):
            yaw = (i / float(frames)) * 2.0 * math.pi
            rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, yaw), 'XYZ').to_quaternion()
            redraw_gui_frame()
        gl_finish()

        t0 = time.perf_counter()
        cpu_draw_sum = 0.0
        for frame_idx in range(frames):
            yaw = (frame_idx / float(frames)) * 2.0 * math.pi
            pitch = math.radians(60.0 + 10.0 * math.sin(2.0 * yaw))
            rv3d.view_rotation = mathutils.Euler((pitch, 0.0, yaw), 'XYZ').to_quaternion()
            t_a = time.perf_counter()
            redraw_gui_frame()
            t_b = time.perf_counter()
            cpu_draw_sum += (t_b - t_a)
        gl_finish()
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed * 1000.0) / frames
        fps = float(frames) / max(elapsed, 1e-6)
        print(f"  [{label:<55}] : {fps:6.1f} FPS ({avg_ms:6.2f} ms/frame)")
        return fps, avg_ms

    results = []

    # 1. Native Baseline (Tessera OFF)
    bpy.ops.tessera.batch_disable_all()
    bpy.context.view_layer.update()
    redraw_gui_frame()
    fps, ms = run_orbit("Native Blender Baseline (Tessera OFF)", frames=20)
    results.append(("Native Blender Baseline", fps, ms))

    # 2. Enable All Tessera
    bpy.ops.tessera.batch_enable_all()
    bpy.context.view_layer.update()
    redraw_gui_frame()

    configs = [
        ("Tessera Default (px=1.0, sc=4.0, dist=10, int=2.5)", 1.0, 4.0, 10.0, 2.5),
        ("Scatter Error = 8.0", 1.0, 8.0, 10.0, 2.5),
        ("Scatter Error = 16.0", 1.0, 16.0, 10.0, 2.5),
        ("Scatter Error = 24.0", 1.0, 24.0, 10.0, 2.5),
        ("Scatter Error = 32.0", 1.0, 32.0, 10.0, 2.5),
        ("LOD Distance = 5.0, Intensity = 5.0, sc=8.0", 1.0, 8.0, 5.0, 5.0),
        ("LOD Distance = 5.0, Intensity = 10.0, sc=16.0", 1.0, 16.0, 5.0, 10.0),
        ("LOD Distance = 3.0, Intensity = 15.0, sc=24.0", 1.0, 24.0, 3.0, 15.0),
        ("LOD Distance = 2.0, Intensity = 20.0, sc=32.0", 1.0, 32.0, 2.0, 20.0),
        ("Coarser Base (px=2.0, sc=16.0, dist=5, int=5)", 2.0, 16.0, 5.0, 5.0),
        ("Aggressive (px=3.0, sc=24.0, dist=3, int=10)", 3.0, 24.0, 3.0, 10.0),
        ("Ultra Fast (px=4.0, sc=32.0, dist=2, int=15)", 4.0, 32.0, 2.0, 15.0),
    ]

    for label, px_err, sc_err, lod_dist, lod_int in configs:
        scene["tessera_batch_pixel_error"] = px_err
        scene["tessera_batch_lod_distance"] = lod_dist
        scene["tessera_batch_lod_intensity"] = lod_int
        scene["tessera_scatter_pixel_error"] = sc_err
        for ob in mesh_objects:
            ob["tessera_pixel_error"] = px_err
            ob["tessera_lod_distance"] = lod_dist
            ob["tessera_lod_intensity"] = lod_int
            ob["tessera_scatter_pixel_error"] = sc_err
        bpy.context.view_layer.update()
        redraw_gui_frame()
        fps, ms = run_orbit(label, frames=25)
        results.append((label, fps, ms))

    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    for label, fps, ms in results:
        print(f"  {label:<55} : {fps:6.1f} FPS ({ms:6.2f} ms)")
    print("=" * 80)

    # Clean exit
    bpy.ops.wm.quit_blender()
    return None

if __name__ == "__main__":
    bpy.app.timers.register(run_sweep, first_interval=0.5)
