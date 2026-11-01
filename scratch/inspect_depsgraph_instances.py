import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()
print(f"Total evaluated object instances in depsgraph: {len(depsgraph.object_instances)}")

instances_by_name = {}
for inst in depsgraph.object_instances:
    name = inst.object.name
    instances_by_name[name] = instances_by_name.get(name, 0) + 1

for name, count in sorted(instances_by_name.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name}: {count} instances (is_instance={inst.is_instance})")

bpy.ops.wm.quit_blender()

