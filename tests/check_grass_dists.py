import bpy
import mathutils
import math

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

center = mathutils.Vector((5.78, 14.30, 1.15))
orbit_dist = 32.67

for name in ["Galetta Grass", "Debris_Leafs", "Rocks_Large", "SweetViolet", "DewBerry", "StickyWilly"]:
    ob = scene.objects.get(name)
    if not ob:
        continue
    inst_positions = []
    for dup in depsgraph.object_instances:
        if dup.parent and dup.parent.original == ob:
            inst_positions.append(dup.matrix_world.translation)
    if not inst_positions:
        continue
    print(f"\nAsset: {name} (total {len(inst_positions)} instances)")
    for f in [0, 15, 30, 45]:
        yaw = (f / 60.0) * 2.0 * math.pi
        pitch = math.radians(60.0 + 10.0 * math.sin(2.0 * yaw))
        rot = mathutils.Euler((pitch, 0.0, yaw), 'XYZ').to_quaternion()
        cam_pos = center + rot @ mathutils.Vector((0.0, 0.0, orbit_dist))
        dists = [(p - cam_pos).length for p in inst_positions]
        dists.sort()
        print(f"  Frame {f:2d}: min={min(dists):.1f}m, median={dists[len(dists)//2]:.1f}m, max={max(dists):.1f}m")
