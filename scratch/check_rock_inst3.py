import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()
count = 0
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        pname = inst.parent.name if inst.parent else None
        loc = tuple(round(x, 2) for x in inst.matrix_world.translation)
        print(f"[{count}] is_instance={inst.is_instance}, parent={pname}, loc={loc}")
        count += 1
        if count >= 10:
            break

bpy.ops.wm.quit_blender()
