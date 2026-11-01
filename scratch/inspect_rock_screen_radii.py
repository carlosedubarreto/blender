import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

depsgraph = bpy.context.evaluated_depsgraph_get()

scales = []
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        s = inst.matrix_world.to_scale()
        scales.append(max(s))

print(f"Total rocks: {len(scales)}")
print(f"Min scale: {min(scales):.4f}, Max scale: {max(scales):.4f}, Avg scale: {sum(scales)/len(scales):.4f}")

# Screen radius in pixels at distance d for rock radius = 0.3 * scale
# Viewport height 1080, projScaleY ~ 1.732
# px_scale = 1.732 * 540 = 935
px_scale = 935.0
mesh_radius = 0.3

screen_radii = []
for inst in depsgraph.object_instances:
    if inst.object.name == 'Rocks_Small':
        cam_p = bpy.data.objects['Camera'].matrix_world.translation # or view
        inst_p = inst.matrix_world.translation
        d = (inst_p - cam_p).length
        s = max(inst.matrix_world.to_scale())
        r_px = (mesh_radius * s * px_scale) / max(d, 0.01)
        screen_radii.append(r_px)

print(f"Min screen radius: {min(screen_radii):.2f} px")
print(f"Max screen radius: {max(screen_radii):.2f} px")
print(f"Avg screen radius: {sum(screen_radii)/len(screen_radii):.2f} px")
print(f"Rocks < 3.0 px: {sum(1 for r in screen_radii if r < 3.0)}")
print(f"Rocks < 4.5 px: {sum(1 for r in screen_radii if r < 4.5)}")
print(f"Rocks < 6.0 px: {sum(1 for r in screen_radii if r < 6.0)}")
print(f"Rocks < 8.0 px: {sum(1 for r in screen_radii if r < 8.0)}")
print(f"Rocks < 10.0 px: {sum(1 for r in screen_radii if r < 10.0)}")

bpy.ops.wm.quit_blender()

