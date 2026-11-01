import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

print("\n--- CHECKING OBJECT PROPERTIES ---")
for ob in bpy.context.scene.objects:
    matching = [k for k in ob.keys() if any(w in k.lower() for w in ['scatter', 'nano', 'anno', 'flow', 'lod', 'proxy', 'tessera'])]
    if matching:
        print(f"Object {ob.name}: {matching}")
        for m in matching:
            print(f"   {m} = {ob[m]}")

print("\n--- CHECKING ADDONS / MODULES ---")
import sys
for mod in sorted(sys.modules.keys()):
    if any(w in mod.lower() for w in ['scatter', 'nano', 'anno', 'tessera']):
        print(f"Module: {mod}")

bpy.ops.wm.quit_blender()

