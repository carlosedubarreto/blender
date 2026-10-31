import bpy

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

for name in ["Galetta Grass", "Debris_Leafs", "Rocks_Small", "DewBerry", "Rocks_Large"]:
    ob = bpy.data.objects.get(name)
    if not ob:
        continue
    eval_ob = ob.evaluated_get(depsgraph)
    # Count instances from depsgraph
    count = 0
    for dup in depsgraph.object_instances:
        if dup.parent == ob:
            count += 1
    print(f"Object: {name}, duplis from depsgraph: {count}")

