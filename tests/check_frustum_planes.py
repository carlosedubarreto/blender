import bpy
import mathutils

cam = bpy.data.objects.get("Camera")
scene = bpy.context.scene

# Get camera matrices
render = scene.render
winx = (render.xsch * render.size) // 100
winy = (render.ysch * render.size) // 100

# Compute camera view and projection matrices
cam_data = cam.data
viewmat = cam.matrix_world.inverted()
winmat = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=winx, y=winy)

viewProj = winmat @ viewmat

print("winmat:", winmat)
print("viewmat:", viewmat)

# Let's extract frustum planes exactly like extract_frustum_planes in tessera_lod.cc:
# Note: Blender mathutils matrices are row-major or col-major?
# In Blender Python: m[row][col].
# But in C++: float4x4 m: m[col][row] or m[row][col]?
print("Let's check C++ float4x4 convention in Blender!")

