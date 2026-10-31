import bpy
import tessera.ui

for ob in bpy.data.objects:
    if ob.type == 'MESH':
        nc2 = tessera.ui.get_existing_nc2_path(ob)
        print(f"MESH: {ob.name:<30} -> NC2: {nc2}")

