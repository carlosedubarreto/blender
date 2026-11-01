import bpy
import mathutils

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
r3d = space.region_3d

# Viewport resolution:
win_region = next(r for r in area.regions if r.type == 'WINDOW')
width = win_region.width
height = win_region.height

print(f"Viewport resolution: {width} x {height}")

# In workbench_engine.cc:
# const float px_scale = (has_cam_lod ? cam_proj_scale_y : viewport_proj_scale_y) * (float(resolution.y) * 0.5f);
# In workbench_engine:
# viewport_persmat = ...
# proj_scale_y is std::abs(winmat[1][1])
# In blender viewport:
# persmat = r3d.perspective_matrix
# winmat = r3d.window_matrix

winmat = r3d.window_matrix
viewport_proj_scale_y = abs(winmat[1][1])
px_scale = viewport_proj_scale_y * (height * 0.5)

print(f"viewport_proj_scale_y = {viewport_proj_scale_y:.4f}")
print(f"px_scale = {px_scale:.4f}")

# Mesh bounds sphere of Rocks_Small:
# In C++:
# bool have_bs = tessera::mesh_bounds_sphere(state.data, field_bs_center, field_bs_radius);
# Let's inspect data.header.boundsMin and boundsMax of Rocks_Small.nc2!
import struct
nc2_path = r"C:\blender-git\tessera_cenas_teste\Rocks_Small.nc2"
with open(nc2_path, "rb") as f:
    hdr = f.read(88)
    bmin = struct.unpack("<fff", hdr[48:60])
    bmax = struct.unpack("<fff", hdr[60:72])
    print("Rocks_Small.nc2 boundsMin:", bmin)
    print("Rocks_Small.nc2 boundsMax:", bmax)

    center = [(bmin[i] + bmax[i])*0.5 for i in range(3)]
    radius = (sum((bmax[i] - bmin[i])**2 for i in range(3)))**0.5 * 0.5
    print(f"center: {center}, radius: {radius}")

depsgraph = bpy.context.evaluated_depsgraph_get()
cam_pos = r3d.view_matrix.inverted().translation

survived = 0
min_screen_radius_px = 4.5
max_draw_dist = 22.0
max_dist_sq = max_draw_dist * max_draw_dist

for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        mat = inst.matrix_world
        # c = (mat * float4(mesh_center, 1.0f)).xyz()
        c = mat @ mathutils.Vector(center)
        d_sq = (c - cam_pos).length_squared
        if max_dist_sq > 0 and d_sq > max_dist_sq:
            continue
        
        # inst_scale_sq
        s0 = mat.col[0].to_3d().length_squared
        s1 = mat.col[1].to_3d().length_squared
        s2 = mat.col[2].to_3d().length_squared
        inst_scale_sq = max(s0, s1, s2)
        r_sq = (radius * radius) * inst_scale_sq
        r_scaled_sq = r_sq * (px_scale * px_scale)
        if r_scaled_sq < (min_screen_radius_px * min_screen_radius_px) * max(d_sq, 0.0001):
            continue
        survived += 1

print(f"Simulated cull survivors: {survived} / 83379")

bpy.ops.wm.quit_blender()

