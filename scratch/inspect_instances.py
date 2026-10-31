import bpy

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

print("=== SCATTER SYSTEM INSTANCE COUNTS ===")
total_instances = 0
for ob in scene.objects:
    if ob.type != 'MESH' or ob.hide_viewport:
        continue
    eval_ob = ob.evaluated_get(depsgraph)
    # Check duplis
    inst_count = 0
    if eval_ob.is_instancer:
        # Count instances in depsgraph
        for dup in depsgraph.object_instances:
            if dup.is_instance and dup.parent == ob:
                inst_count += 1
    print(f"{ob.name}: is_instancer={eval_ob.is_instancer}, dupli count={inst_count}")
    total_instances += inst_count

print(f"Total scattered instances across all emitters: {total_instances}")

