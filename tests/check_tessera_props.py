import bpy

for ob in bpy.context.scene.objects:
    for k in ob.keys():
        if "tessera" in k:
            print(f"Ob: {ob.name} has prop {k} = {ob[k]}")
for k in bpy.context.scene.keys():
    if "tessera" in k:
        print(f"Scene has prop {k} = {bpy.context.scene[k]}")

