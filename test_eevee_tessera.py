import bpy
import os
import sys

def test_eevee_tessera():
    print("[TEST] Setting up EEVEE scene with Tessera...")
    # Clear existing objects without resetting color management
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.use_lock_interface = True

    # Add Suzanne monkey
    bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 0))
    ob = bpy.context.active_object
    ob.name = "Suzanne_Tessera"

    # Assign 2 material slots to test multi-material partitioning
    mat1 = bpy.data.materials.new(name="Mat_Head")
    mat1.use_nodes = True
    mat2 = bpy.data.materials.new(name="Mat_Eyes")
    mat2.use_nodes = True
    ob.data.materials.append(mat1)
    ob.data.materials.append(mat2)

    # Assign second half of faces to material slot 1
    for i, poly in enumerate(ob.data.polygons):
        if i % 2 == 1:
            poly.material_index = 1

    # Unwrap UVs properly
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add a Camera looking directly at Suzanne
    bpy.ops.object.camera_add(location=(0, -3.5, 0.2), rotation=(1.52, 0, 0))
    cam = bpy.context.active_object
    scene.camera = cam

    # Ensure mesh is active for building
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)

    # Build Tessera clusters
    nc2_path = os.path.abspath("test_suzanne_multimat.nc2")
    if os.path.exists(nc2_path):
        os.remove(nc2_path)
    bpy.ops.object.tessera_build(filepath=nc2_path)
    assert os.path.exists(nc2_path)
    nc2_size = os.path.getsize(nc2_path)
    print(f"[TEST] Multi-material NC2 file size: {nc2_size} bytes")

    ob["tessera_enabled"] = 1
    ob["tessera_nc2"] = nc2_path
    ob["tessera_pixel_error"] = 1.0
    ob["tessera_view_source"] = 0

    scene.render.resolution_x = 480
    scene.render.resolution_y = 360

    # Check CLI for specific mode
    target_mode = None
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) > 0:
            target_mode = args[0].lower()

    # Test modes
    all_modes = [
        (0, "diffuse", "test_eevee_tessera_diffuse.png"),
        (1, "clusters", "test_eevee_tessera_clusters.png"),
        (2, "normals", "test_eevee_tessera_normals.png"),
        (3, "tangents", "test_eevee_tessera_tangents.png"),
        (4, "uvs", "test_eevee_tessera_uvs.png"),
    ]

    modes = [m for m in all_modes if target_mode is None or target_mode in (str(m[0]), m[1])]
    if not modes:
        print(f"Unknown mode: {target_mode}")
        return

    for mode_idx, mode_name, filename in modes:
        ob["tessera_visualize_clusters"] = mode_idx
        ob.update_tag()
        out_path = os.path.abspath(filename)
        scene.render.filepath = out_path
        print(f"[TEST] Rendering {mode_name.upper()} mode to {out_path}...")
        bpy.ops.render.render(write_still=True)
        assert os.path.exists(out_path)
        print(f"[TEST SUCCESS] {mode_name.capitalize()} image saved: {os.path.getsize(out_path)} bytes")

    print("[ALL TESTS COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    test_eevee_tessera()
