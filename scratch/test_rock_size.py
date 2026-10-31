import bpy
import mathutils

window = bpy.context.window_manager.windows[0]
for area in window.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces.active
        rv3d = space.region_3d
        break

cam_pos = rv3d.view_matrix.inverted().translation
proj_scale_y = rv3d.window_matrix[1][1]
vp_h = 1080.0

print(f"Camera position: {cam_pos}, proj_scale_y: {proj_scale_y}")

# Test for Rocks_Small
# Rocks_Small instances
# We can read them or simulate bounding radius ~0.05m
rock_ob = bpy.data.objects.get("Rocks_Small")
mesh_r = 0.08 # average rock radius ~8cm

# If distance is 30m:
d = 30.0
r_px = (mesh_r * proj_scale_y * (vp_h * 0.5)) / d
print(f"At 30m distance, 8cm rock screen radius: {r_px:.2f} pixels")
d = 50.0
r_px = (mesh_r * proj_scale_y * (vp_h * 0.5)) / d
print(f"At 50m distance, 8cm rock screen radius: {r_px:.2f} pixels")
d = 15m = 15.0
r_px = (mesh_r * proj_scale_y * (vp_h * 0.5)) / 15.0
print(f"At 15m distance, 8cm rock screen radius: {r_px:.2f} pixels")

