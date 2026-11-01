import bpy
import time
import mathutils

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
region = next(r for r in area.regions if r.type == 'WINDOW')
r3d = space.region_3d

ctx = {
    'window': win,
    'screen': win.screen,
    'area': area,
    'region': region,
    'scene': bpy.context.scene,
    'space_data': space,
}

# Warmup
for _ in range(5):
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

print("\n--- TEST: ONLY ROTATE CAMERA, MEASURE REDRAW ---")
rot_step = mathutils.Quaternion((0.0, 0.0, 1.0), 0.017)
for i in range(5):
    t0 = time.perf_counter()
    r3d.view_rotation = rot_step @ r3d.view_rotation
    t1 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t2 = time.perf_counter()
    print(f"Step {i}: rot_time={(t1-t0)*1000:.3f} ms, redraw_time={(t2-t1)*1000:.3f} ms")

bpy.ops.wm.quit_blender()

