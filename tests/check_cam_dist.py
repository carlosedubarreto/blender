import bpy
import mathutils

cam = bpy.data.objects.get("Camera")
print("Camera location:", cam.location)
print("Camera matrix translation:", cam.matrix_world.translation)

# Check distance of instances to camera
depsgraph = bpy.context.evaluated_depsgraph_get()
cam_pos = cam.matrix_world.translation

dists = []
for inst in depsgraph.object_instances:
    if inst.is_instance:
        pos = inst.matrix_world.translation
        dists.append((pos - cam_pos).length)

if dists:
    print(f"Min distance: {min(dists):.2f}m")
    print(f"Max distance: {max(dists):.2f}m")
    print(f"Average distance: {sum(dists)/len(dists):.2f}m")
    print(f"Count < 22m: {sum(1 for d in dists if d < 22.0)}")
    print(f"Count < 28m: {sum(1 for d in dists if d < 28.0)}")
    print(f"Count < 35m: {sum(1 for d in dists if d < 35.0)}")
    print(f"Count >= 35m: {sum(1 for d in dists if d >= 35.0)}")

