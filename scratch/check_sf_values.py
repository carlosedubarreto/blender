import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
area = next(a for a in wm.windows[0].screen.areas if a.type == 'VIEW_3D')
shading = area.spaces.active.shading

print("wm.scatterflow_nano:", getattr(wm, "scatterflow_nano", None))
print("wm.scatterflow_nano_modifiers:", getattr(wm, "scatterflow_nano_modifiers", None))
print("shading.scatterflow_nano_error:", getattr(shading, "scatterflow_nano_error", None))
print("shading.scatterflow_nano_cull_size:", getattr(shading, "scatterflow_nano_cull_size", None))
print("shading.scatterflow_nano_debug:", getattr(shading, "scatterflow_nano_debug", None))

bpy.ops.wm.quit_blender()

