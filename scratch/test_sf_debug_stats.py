import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
area = next(a for a in wm.windows[0].screen.areas if a.type == 'VIEW_3D')
shading = area.spaces.active.shading
shading.scatterflow_nano_debug = 'CLUSTERS'

region = next(r for r in area.regions if r.type == 'WINDOW')
ctx = {
    'window': wm.windows[0],
    'screen': wm.windows[0].screen,
    'area': area,
    'region': region,
    'scene': bpy.context.scene,
    'space_data': area.spaces.active,
}

print("\n--- REDRAWING WITH CLUSTERS DEBUG ---")
with bpy.context.temp_override(**ctx):
    region.tag_redraw()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

bpy.ops.wm.quit_blender()

