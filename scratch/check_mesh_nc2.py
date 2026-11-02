import bpy

for mesh in bpy.data.meshes:
    nc2 = mesh.get('tessera_nc2') or mesh.get('tessera_nc2_path')
    if nc2:
        print(f"Mesh {mesh.name} has nc2: {nc2}")

