import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
area = next(a for a in wm.windows[0].screen.areas if a.type == 'VIEW_3D')
shading = area.spaces.active.shading
prop = shading.bl_rna.properties['scatterflow_nano_debug']
print("Debug enum items:")
for item in prop.enum_items:
    print(f"  {item.identifier}: {item.name} - {item.description}")

# Let's also check scatterflow_nano_modifiers
prop_mod = wm.bl_rna.properties['scatterflow_nano_modifiers']
print("Modifiers prop:", prop_mod.description)

bpy.ops.wm.quit_blender()

