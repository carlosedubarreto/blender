import bpy
import mathutils

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

cam = bpy.context.scene.camera
cam_pos = cam.matrix_world.translation if cam else mathutils.Vector((0, 0, 0))
print("Camera position:", cam_pos)

depsgraph = bpy.context.evaluated_depsgraph_get()
rocks_dists = []
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        pos = inst.matrix_world.translation
        d = (pos - cam_pos).length
        rocks_dists.append(d)

if rocks_dists:
    rocks_dists.sort()
    print(f"Total Rocks_Small instances: {len(rocks_dists)}")
    print(f"Min dist: {rocks_dists[0]:.2f}m")
    print(f"Max dist: {rocks_dists[-1]:.2f}m")
    print(f"Instances < 15m: {sum(1 for d in rocks_dists if d < 15)}")
    print(f"Instances < 25m: {sum(1 for d in rocks_dists if d < 25)}")
    print(f"Instances < 40m: {sum(1 for d in rocks_dists if d < 40)}")
    print(f"Instances < 60m: {sum(1 for d in rocks_dists if d < 60)}")
    print(f"Instances >= 60m: {sum(1 for d in rocks_dists if d >= 60)}")

bpy.ops.wm.quit_blender()

