import bpy
import mathutils

for ob in bpy.data.objects:
    if "ground" in ob.name.lower() or "gravel" in ob.name.lower():
        bbox = [ob.matrix_world @ mathutils.Vector(b) for b in ob.bound_box]
        min_z = min(p.z for p in bbox)
        max_z = max(p.z for p in bbox)
        min_x = min(p.x for p in bbox)
        max_x = max(p.x for p in bbox)
        min_y = min(p.y for p in bbox)
        max_y = max(p.y for p in bbox)
        print(f"Ground '{ob.name}': X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")

