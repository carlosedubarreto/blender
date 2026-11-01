import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()
print(f"Total evaluated object instances: {len(depsgraph.object_instances)}")

counts = {}
for inst in depsgraph.object_instances:
    ob_name = inst.object.name
    counts[ob_name] = counts.get(ob_name, 0) + 1

for name, count in sorted(counts.items(), key=lambda x: -x[1])[:25]:
    print(f"  {name}: {count}")

bpy.ops.wm.quit_blender()

