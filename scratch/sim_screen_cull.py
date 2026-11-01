import bpy
import mathutils
import math

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

cam = bpy.context.scene.camera
cam_pos = cam.matrix_world.translation if cam else mathutils.Vector((0, 0, 0))

depsgraph = bpy.context.evaluated_depsgraph_get()

# Mesh radius of Desert Rock2_V1 is ~0.3m
mesh_radius = 0.3
px_scale = 1300.0 # ~1080p

for threshold in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    survivors = 0
    total = 0
    for inst in depsgraph.object_instances:
        if inst.object.name == 'Rocks_Small':
            total += 1
            pos = inst.matrix_world.translation
            d = (pos - cam_pos).length
            s = inst.matrix_world.to_scale()
            scale = max(s.x, s.y, s.z)
            world_radius = mesh_radius * scale
            screen_r = (world_radius * px_scale) / max(d, 0.01)
            if screen_r >= threshold:
                survivors += 1
    print(f"Threshold {threshold:.1f}px: {survivors} / {total} survivors ({survivors/total*100:.1f}%)")

bpy.ops.wm.quit_blender()

