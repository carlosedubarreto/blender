import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

for ob in bpy.context.scene.objects:
    if 'tessera_stat_drawn_instances' in ob:
        print(f"Object: {ob.name}, drawn: {ob['tessera_stat_drawn_instances']}")

bpy.ops.wm.quit_blender()

