import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        print(f"Rocks_Small: is_instance={inst.is_instance}, is_eval={inst.is_eval_object}, parent={inst.parent.name if inst.parent else None}")
        break

bpy.ops.wm.quit_blender()

