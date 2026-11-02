import bpy
import math
import mathutils

cam = bpy.data.objects.get("Camera")
print("Camera location:", cam.matrix_world.translation)
print("Camera rotation (euler deg):", [math.degrees(a) for a in cam.matrix_world.to_euler()])

# Forward vector of camera (in Blender, camera points along local -Z)
fwd = cam.matrix_world.to_3x3() @ mathutils.Vector((0, 0, -1))
print("Camera forward vector:", fwd)
up = cam.matrix_world.to_3x3() @ mathutils.Vector((0, 1, 0))
print("Camera up vector:", up)
right = cam.matrix_world.to_3x3() @ mathutils.Vector((1, 0, 0))
print("Camera right vector:", right)
