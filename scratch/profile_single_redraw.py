import bpy
import time
import cProfile
import pstats
import io

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

print("\n--- PROFILING ONE REDRAW ---")
pr = cProfile.Profile()
pr.enable()
with bpy.context.temp_override(**ctx):
    region.tag_redraw()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())

bpy.ops.wm.quit_blender()

