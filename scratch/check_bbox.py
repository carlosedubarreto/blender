import bpy
import mathutils

cam = bpy.context.scene.camera
print(f"Camera: {cam.name if cam else None}")
for name in ['Ground_005', 'Gravel_13', 'Ground_013', 'Ground_009', 'Ground_001']:
    ob = bpy.data.objects.get(name)
    if ob:
        bbox = [ob.matrix_world @ mathutils.Vector(b) for b in ob.bound_box]
        min_z = min(b.z for b in bbox)
        max_z = max(b.z for b in bbox)
        min_x = min(b.x for b in bbox)
        max_x = max(b.x for b in bbox)
        min_y = min(b.y for b in bbox)
        max_y = max(b.y for b in bbox)
        print(f"{name}: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")
