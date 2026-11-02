import bpy

depsgraph = bpy.context.evaluated_depsgraph_get()
count = 0
for inst in depsgraph.object_instances:
    if inst.is_instance:
        count += 1
        if count <= 5:
            print(f"Inst {count}: parent={inst.parent.name if inst.parent else 'None'}, ob={inst.instance_object.name if inst.instance_object else 'None'}")
            print("  matrix_world:\n", inst.matrix_world)
            print("  translation:", inst.matrix_world.translation)

