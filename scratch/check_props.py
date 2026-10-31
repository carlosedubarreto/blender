import bpy

for ob in bpy.data.objects:
    if "Rock" in ob.name or "Sweet" in ob.name or "Dew" in ob.name or "Terrain" in ob.name:
        props = {k: ob[k] for k in ob.keys()}
        print(f"Object {ob.name}: props = {props}")

