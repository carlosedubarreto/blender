import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()

cam = bpy.data.objects.get("Camera")
cam_pos = cam.matrix_world.translation
cam_fwd = (cam.matrix_world.to_3x3() @ mathutils.Vector((0, 0, -1))).normalized()

print(f"Camera pos: {cam_pos}, forward: {cam_fwd}")

by_emitter = {}
for inst in depsgraph.object_instances:
    if inst.is_instance and inst.parent:
        name = inst.parent.name
        if name not in by_emitter:
            by_emitter[name] = []
        by_emitter[name].append(inst.matrix_world.translation.copy())

for name, pts in sorted(by_emitter.items()):
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    in_front = sum(1 for p in pts if (p - cam_pos).dot(cam_fwd) > 0)
    print(f"\nEmitter '{name}': {len(pts)} instances")
    print(f"   X range: [{min(xs):.2f}, {max(xs):.2f}]")
    print(f"   Y range: [{min(ys):.2f}, {max(ys):.2f}]")
    print(f"   Z range: [{min(zs):.2f}, {max(zs):.2f}]")
    print(f"   In front of camera: {in_front} / {len(pts)}")
