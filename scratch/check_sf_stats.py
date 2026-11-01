import bpy

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

with bpy.context.temp_override(**ctx):
    region.tag_redraw()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

# Check info overlay in window / areas / header
print("Checking info string:")
# In bpy, we can check space.overlay or scene stats
for line in bpy.context.scene.statistics(bpy.context.view_layer).split('|'):
    print(" ", line.strip())

bpy.ops.wm.quit_blender()

