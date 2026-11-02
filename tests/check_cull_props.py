import bpy

print("=== CHECKING CUSTOM PROPERTIES ===")
for ob in bpy.data.objects:
    for k in ob.keys():
        if "cull" in k.lower() or "cam" in k.lower() or "view" in k.lower() or "lod" in k.lower() or "tessera" in k.lower():
            print(f"Object '{ob.name}' property '{k}' = {ob[k]}")

for k in bpy.context.scene.keys():
    if "cull" in k.lower() or "cam" in k.lower() or "view" in k.lower() or "lod" in k.lower() or "tessera" in k.lower():
        print(f"Scene property '{k}' = {bpy.context.scene[k]}")

