import bpy
import time

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
region = next(r for r in area.regions if r.type == 'WINDOW')

ctx = {
    'window': win,
    'screen': win.screen,
    'area': area,
    'region': region,
    'scene': bpy.context.scene,
    'space_data': space,
}

print("\n--- MEASURING 10 STATIONARY FRAMES ---")
for i in range(10):
    t0 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    print(f"Stationary Frame {i}: {(t1-t0)*1000:.2f} ms")

print("\n--- MEASURING 10 ORBIT FRAMES (view3d.orbit operator) ---")
for i in range(10):
    t0 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        bpy.ops.view3d.orbit(angle=0.05, type='ORBITRIGHT')
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    print(f"Orbit Frame {i}: {(t1-t0)*1000:.2f} ms")

bpy.ops.wm.quit_blender()

