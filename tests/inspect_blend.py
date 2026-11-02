import bpy
import mathutils

# Let's inspect an instance from DewBerry or SweetViolet
for ob in bpy.data.objects:
    if ob.name in ("SweetViolet", "DewBerry"):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_ob = ob.evaluated_get(depsgraph)
        print(f"Eval ob: {eval_ob.name}")
        instances = [inst for inst in depsgraph.object_instances if inst.parent and inst.parent.original == ob]
        print(f"Found {len(instances)} instances for {ob.name}")
        if instances:
            for i in range(min(5, len(instances))):
                print(f"  inst {i} pos: {instances[i].matrix_world.to_translation()}")
