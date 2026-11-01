import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()
scales = []
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        s = inst.matrix_world.to_scale()
        scales.append((s.x + s.y + s.z) / 3.0)

if scales:
    scales.sort()
    print(f"Total: {len(scales)}")
    print(f"Min scale: {scales[0]:.4f}, Max scale: {scales[-1]:.4f}, Median: {scales[len(scales)//2]:.4f}")

bpy.ops.wm.quit_blender()

