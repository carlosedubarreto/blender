import bpy

bpy.ops.wm.open_mainfile(filepath=r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend")
for ob in bpy.context.scene.objects:
    if 'tessera_enabled' in ob:
        print(f"{ob.name}: tessera_enabled = {ob['tessera_enabled']}")
bpy.ops.wm.quit_blender()

