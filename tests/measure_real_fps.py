import bpy
import time
import math
import mathutils
import gpu

def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return window, area, region, area.spaces.active, area.spaces.active.region_3d
    return None, None, None, None, None

def run_bench():
    window, area, region, space, rv3d = find_view3d()
    if not area or not rv3d:
        return 0.1
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    space.shading.type = 'SOLID'
    space.overlay.show_overlays = True
    rv3d.view_perspective = 'PERSP'

    mesh_objects = [ob for ob in scene.objects if ob.type == 'MESH']
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

    # Enable Tessera everywhere
    bpy.ops.tessera.batch_enable_all()
    bpy.context.view_layer.update()

    def run_pass(mode_type, label, frames=30):
        # Warmup
        for i in range(3):
            yaw = (i / float(frames)) * 2.0 * math.pi
            rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, yaw), 'XYZ').to_quaternion()
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type=mode_type, iterations=1)
        gpu.state.finish()

        t0 = time.perf_counter()
        cpu_t = 0.0
        gpu_t = 0.0
        for i in range(frames):
            yaw = (i / float(frames)) * 2.0 * math.pi
            pitch = math.radians(60.0 + 10.0 * math.sin(2.0 * yaw))
            rv3d.view_rotation = mathutils.Euler((pitch, 0.0, yaw), 'XYZ').to_quaternion()
            
            t_a = time.perf_counter()
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type=mode_type, iterations=1)
            t_b = time.perf_counter()
            gpu.state.finish()
            t_c = time.perf_counter()
            cpu_t += (t_b - t_a)
            gpu_t += (t_c - t_b)

        el = time.perf_counter() - t0
        ms = (el * 1000.0) / frames
        cpu_ms = (cpu_t * 1000.0) / frames
        gpu_ms = (gpu_t * 1000.0) / frames
        fps = float(frames) / max(el, 1e-6)
        print(f"\n[{label}] : {fps:6.1f} FPS (total: {ms:6.2f} ms | cpu_redraw: {cpu_ms:6.2f} ms, gpu_finish: {gpu_ms:6.2f} ms)")
        return fps, ms

    run_pass('DRAW_WIN_SWAP', "ALL TESSERA (DRAW_WIN_SWAP)")
    run_pass('DRAW_SWAP',     "ALL TESSERA (DRAW_SWAP)")
    run_pass('DRAW',          "ALL TESSERA (PURE DRAW)")

    bpy.ops.wm.quit_blender()
    return None

bpy.app.timers.register(run_bench, first_interval=0.5)

