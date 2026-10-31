import bpy

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

ob = bpy.data.objects.get("Galetta Grass")
if ob:
    eval_ob = ob.evaluated_get(depsgraph)
    mesh = eval_ob.to_mesh()
    print(f"Galetta Grass evaluated mesh: {len(mesh.vertices):,} vertices, {len(mesh.polygons):,} polygons")
    ob.to_mesh_clear()

    # Check Geometry Nodes modifier
    for m in ob.modifiers:
        if m.type == 'NODES':
            print(f"Modifier: {m.name}, node_group: {m.node_group.name}")

