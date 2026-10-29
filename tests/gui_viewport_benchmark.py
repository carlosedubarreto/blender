"""Live Interactive GUI Viewport Orbit Benchmark for Tessera_HighPoy_Test_1.blend.

Must be run in real GUI mode (NO -b flag):
    "C:\\blender-git\\build_windows_x64_vc17_Release\\bin\\blender.exe" "D:\\_Code\\Meu\\Nanite\\Tessera_HighPoy_Test_1.blend" --python gui_viewport_benchmark.py
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


def run_benchmark():
    if bpy.app.background:
        print("FATAL: Benchmark must be run in interactive GUI mode (without -b)!")
        sys.exit(1)

    window, area, region, space, rv3d = find_view3d()
    if area is None or rv3d is None:
        return 0.1  # Wait for GUI window initialization

    exit_code = 0
    try:
        print("\n" + "=" * 80)
        print("LIVE INTERACTIVE GUI VIEWPORT BENCHMARK (USER PERSPECTIVE - SOLID MODE)")
        print(f"Binary : {bpy.app.binary_path}")
        print(f"File   : {bpy.data.filepath}")
        try:
            import gpu
            print(f"GPU    : {gpu.platform.backend_type_get()} | {gpu.platform.device_type_get()} | {gpu.platform.renderer_get()}")
        except Exception as e:
            print(f"GPU    : {e}")
        print("=" * 80)

        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_WORKBENCH'
        space.shading.type = 'SOLID'
        rv3d.view_perspective = 'PERSP'  # Explicitly outside of Camera View (User Perspective)

        mesh_objects = [ob for ob in scene.objects if ob.type == 'MESH']
        print(f"Detected {len(mesh_objects)} mesh objects in scene:")
        for ob in mesh_objects:
            v_cnt = len(ob.data.vertices) if ob.data else 0
            p_cnt = len(ob.data.polygons) if ob.data else 0
            mods = [m.type for m in ob.modifiers]
            print(f"  - {ob.name}: {v_cnt:,} verts, {p_cnt:,} polys, mods={mods}, hide_vp={ob.hide_viewport}")

        # Compute scene bounding center and orbit distance
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

        def run_orbit_60(label):
            # Warmup 5 frames
            for i in range(5):
                yaw = (i / 60.0) * 2.0 * math.pi
                rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, yaw), 'XYZ').to_quaternion()
                redraw_gui_frame()
            gl_finish()

            t0 = time.perf_counter()
            cpu_draw_sum = 0.0
            gl_finish_sum = 0.0
            for frame_idx in range(60):
                yaw = (frame_idx / 60.0) * 2.0 * math.pi
                pitch = math.radians(60.0 + 10.0 * math.sin(2.0 * yaw))
                rv3d.view_rotation = mathutils.Euler((pitch, 0.0, yaw), 'XYZ').to_quaternion()
                t_a = time.perf_counter()
                redraw_gui_frame()
                t_b = time.perf_counter()
                gl_finish()
                t_c = time.perf_counter()
                cpu_draw_sum += (t_b - t_a)
                gl_finish_sum += (t_c - t_b)
            elapsed = time.perf_counter() - t0
            avg_ms = (elapsed * 1000.0) / 60.0
            cpu_ms = (cpu_draw_sum * 1000.0) / 60.0
            gpu_wait_ms = (gl_finish_sum * 1000.0) / 60.0
            fps = 60.0 / max(elapsed, 1e-6)
            print(f"  [{label:<42}] : {fps:6.1f} FPS ({avg_ms:6.2f} ms/frame | redraw={cpu_ms:6.2f}ms, glFinish={gpu_wait_ms:6.2f}ms)")
            return fps, avg_ms

        # -------------------------------------------------------------------------
        # Phase 1: Native Blender Baseline (Tessera OFF) in User Perspective
        # -------------------------------------------------------------------------
        for ob in mesh_objects:
            ob["tessera_enabled"] = 0
            if hasattr(ob, "tessera"):
                ob.tessera.enabled = False
        bpy.context.view_layer.update()
        redraw_gui_frame()

        base_fps, base_ms = run_orbit_60("Native Blender Baseline (Tessera OFF)")

        # -------------------------------------------------------------------------
        # Phase 2: Tessera Enabled (Full Base Mesh + Indexed Hardware Indirect)
        # -------------------------------------------------------------------------
        import tessera.ui
        for ob in mesh_objects:
            nc2 = tessera.ui.get_existing_nc2_path(ob)
            if nc2:
                ob["tessera_nc2"] = nc2
                ob["tessera_nc2_path"] = nc2
                sz_mb = os.path.getsize(nc2) / (1024 * 1024)
                print(f"  -> {ob.name} nc2: {nc2} ({sz_mb:.2f} MB)")
            ob["tessera_enabled"] = 1
            ob["tessera_visualize_clusters"] = 0
            ob["tessera_pixel_error"] = 1.0
            ob["tessera_culling_margin"] = 0.02
            if hasattr(ob, "tessera"):
                ob.tessera.enabled = True
        bpy.context.view_layer.update()
        redraw_gui_frame()

        tess_fps, tess_ms = run_orbit_60("Tessera ON (Solid Diffuse, User Persp)")

        # -------------------------------------------------------------------------
        # Phase 3: Tessera Enabled (Cluster Color Visualization Mode)
        # -------------------------------------------------------------------------
        for ob in mesh_objects:
            ob["tessera_visualize_clusters"] = 1
        bpy.context.view_layer.update()
        redraw_gui_frame()

        vis_fps, vis_ms = run_orbit_60("Tessera ON (Cluster Colors, User Persp)")

        for ob in mesh_objects:
            ob["tessera_visualize_clusters"] = 0

        # -------------------------------------------------------------------------
        # Phase 4: User Perspective Frustum Culling & Telemetry Inspection
        # -------------------------------------------------------------------------
        print("\nVerifying User Perspective Frustum Culling (On-Screen vs Off-Screen):")
        scene["tessera_navigating"] = 0
        for ob in mesh_objects:
            ob["tessera_show_stats"] = 1

        # Look directly at center (wait 250ms so navigation cooldown expires and stats update)
        rv3d.view_location = center
        rv3d.view_distance = orbit_dist
        rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, 0.0), 'XYZ').to_quaternion()
        time.sleep(0.25)
        redraw_gui_frame()
        redraw_gui_frame()
        for ob in mesh_objects:
            print(
                f"  [Stats {ob.name}] clusters={ob.get('tessera_stat_clusters', -1)}/{ob.get('tessera_total_clusters', -1)}, "
                f"tris={ob.get('tessera_stat_triangles', -1)}/{ob.get('tessera_total_triangles', -1)}, "
                f"lod={ob.get('tessera_stat_lod_min', -1)}..{ob.get('tessera_stat_lod_max', -1)}, "
                f"dispatch={ob.get('tessera_stat_dispatch_ms', -1):.3f}ms, "
                f"trav={ob.get('tessera_stat_gpu_traverse_ms', -1):.3f}ms, "
                f"rast={ob.get('tessera_stat_gpu_raster_ms', -1):.3f}ms"
            )
        onscreen_clusters = sum(int(ob.get("tessera_stat_clusters", 0)) for ob in mesh_objects)
        onscreen_tris = sum(int(ob.get("tessera_stat_triangles", 0)) for ob in mesh_objects)

        # Pan User Perspective far away so all meshes are outside the User Perspective frustum
        rv3d.view_location = center + mathutils.Vector((diag * 20.0, diag * 20.0, 0.0))
        rv3d.view_distance = 2.0
        time.sleep(0.25)
        redraw_gui_frame()
        redraw_gui_frame()
        offscreen_clusters = sum(int(ob.get("tessera_stat_clusters", 0)) for ob in mesh_objects)
        offscreen_tris = sum(int(ob.get("tessera_stat_triangles", 0)) for ob in mesh_objects)

        print(f"  User Perspective On-Screen  : {onscreen_clusters:,} clusters ({onscreen_tris:,} triangles)")
        print(f"  User Perspective Off-Screen : {offscreen_clusters:,} clusters ({offscreen_tris:,} triangles)")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Native Baseline (Tessera OFF) : {base_fps:6.1f} FPS ({base_ms:6.2f} ms)")
        print(f"  Tessera ON (Solid Diffuse)    : {tess_fps:6.1f} FPS ({tess_ms:6.2f} ms)")
        print(f"  Tessera ON (Cluster Colors)   : {vis_fps:6.1f} FPS ({vis_ms:6.2f} ms)")
        print("=" * 80)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        bpy.ops.wm.quit_blender()
        if exit_code != 0:
            sys.exit(exit_code)
    return None


bpy.app.timers.register(run_benchmark, first_interval=0.5)
