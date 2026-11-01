import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

ob = bpy.data.objects['Rocks_Small']
print("Rocks_Small dimensions:", ob.dimensions)
mesh = ob.data
print(f"Vertices: {len(mesh.vertices)}, Polygons: {len(mesh.polygons)}")

# Check children or archetypes
for child in ob.children:
    print(f"Child: {child.name}, dims: {child.dimensions}")

bpy.ops.wm.quit_blender()

