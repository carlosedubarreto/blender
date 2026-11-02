import bpy
import mathutils

cam = bpy.data.objects.get("Camera")
scene = bpy.context.scene

render = scene.render
winx = (render.resolution_x * render.resolution_percentage) // 100
winy = (render.resolution_y * render.resolution_percentage) // 100

cam_data = cam.data
viewmat = cam.matrix_world.inverted()
winmat = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=winx, y=winy)

print("cam clip_start:", cam_data.clip_start, "clip_end:", cam_data.clip_end)
print("winmat:\n", winmat)

# Let's project points in camera space (where camera looks along -Z):
# Point on near plane: (0, 0, -clip_start)
p_near = mathutils.Vector((0, 0, -cam_data.clip_start, 1.0))
p_mid = mathutils.Vector((0, 0, -(cam_data.clip_start + cam_data.clip_end)*0.5, 1.0))
p_far = mathutils.Vector((0, 0, -cam_data.clip_end, 1.0))

c_near = winmat @ p_near
c_mid = winmat @ p_mid
c_far = winmat @ p_far

print("c_near:", c_near, "ndc_z:", c_near.z / c_near.w)
print("c_mid: ", c_mid, "ndc_z:", c_mid.z / c_mid.w)
print("c_far: ", c_far, "ndc_z:", c_far.z / c_far.w)
