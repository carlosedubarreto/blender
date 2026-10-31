import ctypes
import math
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

def run_isolation():
    if bpy.app.background:
        return 0.1
    window, area, region, space, rv3d = find_view3d()
    if area is None or rv3d is None:
        return 0.1

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    space.shading.type = 'SOLID'
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

    def redraw():
        with bpy.context.temp_override(**override):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def measure(label, frames=20):
        for i in range(3):
            yaw = (i / float(frames)) * 2.0 * math.pi
            rv3d.view_rotation = mathutils.Euler((math.radians(65.0), 0.0, yaw), 'XYZ').to_quaternion()
            redraw()
        gl_finish()
        t0 = time.perf_counter()
        for i in range(frames):
            yaw = (i / float(frames)) * 2.0 * math.pi
            pitch = math.radians(60.0 + 10.0 * math.sin(2.0 * yaw))
            rv3d.view_rotation = mathutils.Euler((pitch, 0.0, yaw), 'XYZ').to_quaternion()
            redraw()
        gl_finish()
        el = time.perf_counter() - t0
        ms = (el * 1000.0) / frames
        fps = float(frames) / max(el, 1e-6)
        print(f"  [{label:<45}] : {fps:6.1f} FPS ({ms:6.2f} ms)")
        return fps, ms

    # First enable all Tessera
    bpy.ops.tessera.batch_enable_all()
    bpy.context.view_layer.update()
    redraw()

    measure("ALL Tessera Objects ON")

    # Isolate groups
    groups = {
        "ONLY Ground & Gravel": lambda name: "Ground" in name or "Gravel" in name,
        "ONLY Galetta Grass": lambda name: "Galetta" in name or "Grass" in name,
        "ONLY SweetViolet": lambda name: "SweetViolet" in name,
        "ONLY DewBerry": lambda name: "DewBerry" in name,
        "ONLY StickyWilly": lambda name: "StickyWilly" in name,
        "ONLY Rocks_Large": lambda name: "Rocks_Large" in name or "Desert Rock2" in name,
        "ONLY Rocks_Small": lambda name: "Rocks_Small" in name or "Desert Rock3" in name,
        "ONLY Debris_Leafs": lambda name: "Debris" in name or "Leaf" in name,
        "ALL EXCEPT Galetta Grass": lambda name: "Galetta" not in name and "Grass" not in name,
        "ALL EXCEPT Debris_Leafs": lambda name: "Debris" not in name and "Leaf" not in name,
        "ALL EXCEPT Grass & Debris": lambda name: ("Galetta" not in name and "Grass" not in name and "Debris" not in name and "Leaf" not in name),
    }

    for grp_name, predicate in groups.items():
        for ob in mesh_objects:
            enabled = 1 if predicate(ob.name) else 0
            ob["tessera_enabled"] = enabled
            if hasattr(ob, "tessera"):
                ob.tessera.enabled = bool(enabled)
            # hide native viewport if tessera is disabled so native doesn't draw it!
            ob.hide_viewport = not bool(enabled)
        bpy.context.view_layer.update()
        redraw()
        measure(grp_name)

    bpy.ops.wm.quit_blender()
    return None

if __name__ == "__main__":
    bpy.app.timers.register(run_isolation, first_interval=0.5)

