import bpy
import mathutils

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
r3d = space.region_3d

print("View matrix location (cam pos):", r3d.view_matrix.inverted().translation)
print("View perspective:", r3d.view_perspective)
print("View distance:", r3d.view_distance)
print("View camera:", space.camera.name if space.camera else None)
if space.camera:
    cam_ob = space.camera
    print("Camera world loc:", cam_ob.matrix_world.translation)

depsgraph = bpy.context.evaluated_depsgraph_get()

rock_dists = []
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        cam_p = r3d.view_matrix.inverted().translation
        inst_p = inst.matrix_world.translation
        d = (inst_p - cam_p).length
        rock_dists.append(d)

print(f"Total rocks: {len(rock_dists)}")
if rock_dists:
    print(f"Min rock dist: {min(rock_dists):.2f}m")
    print(f"Max rock dist: {max(rock_dists):.2f}m")
    print(f"Avg rock dist: {sum(rock_dists)/len(rock_dists):.2f}m")
    print(f"Rocks < 10m: {sum(1 for d in rock_dists if d < 10)}")
    print(f"Rocks < 20m: {sum(1 for d in rock_dists if d < 20)}")
    print(f"Rocks < 30m: {sum(1 for d in rock_dists if d < 30)}")
    print(f"Rocks < 50m: {sum(1 for d in rock_dists if d < 50)}")
    print(f"Rocks >= 50m: {sum(1 for d in rock_dists if d >= 50)}")

bpy.ops.wm.quit_blender()

