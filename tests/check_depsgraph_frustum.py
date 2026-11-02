import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()
cam = bpy.data.objects.get("Camera")
print(f"Scene active camera: {cam.name if cam else 'None'}")

# Let's inspect the evaluated instances of Debris_Leafs or other scatter objects
for ob in bpy.data.objects:
    if "scatter" in ob.name.lower() or "debris" in ob.name.lower() or "grass" in ob.name.lower() or "violet" in ob.name.lower():
        eval_ob = ob.evaluated_get(depsgraph)
        print(f"\nObject: {ob.name}, type: {ob.type}")
        # Check modifiers
        for m in ob.modifiers:
            if m.type == 'NODES':
                print(f"  Modifier: {m.name}, show_viewport={m.show_viewport}, show_render={m.show_render}")

# Now let's check all depsgraph instances and test their positions against the Camera frustum
if cam:
    cam_eval = cam.evaluated_get(depsgraph)
    # Get camera view_frame or projection
    scene = bpy.context.scene
    frame = cam.data.view_frame(scene=scene)
    print("Camera frame points (local):", frame)
    cam_mat = cam_eval.matrix_world
    inv_cam_mat = cam_mat.inverted()

    # Collect instance locations from depsgraph
    inside_count = 0
    outside_count = 0
    total_inst = 0

    # Let's sample instances from depsgraph
    for inst in depsgraph.object_instances:
        if inst.is_instance:
            total_inst += 1
            # instance world matrix
            mat = inst.matrix_world
            pos = mat.translation
            # transform to camera space
            cam_p = inv_cam_mat @ pos
            # In Blender camera space:
            # -Z is looking forward (cam_p.z < 0)
            # Clip planes:
            if cam_p.z >= 0:
                outside_count += 1
                continue
            dist = -cam_p.z
            # Perspective projection bounds:
            # frame corners give slope:
            # corner 0 has local x, y, z=-1
            # slope_x = abs(frame[0].x / frame[0].z), slope_y = abs(frame[0].y / frame[0].z)
            slope_x = abs(frame[0].x / frame[0].z)
            slope_y = abs(frame[0].y / frame[0].z)
            in_x = abs(cam_p.x) <= slope_x * dist
            in_y = abs(cam_p.y) <= slope_y * dist
            in_z = (dist >= cam.data.clip_start) and (dist <= cam.data.clip_end)
            if in_x and in_y and in_z:
                inside_count += 1
            else:
                outside_count += 1

    print(f"\nDEPSGRAPH INSTANCES SUMMARY:")
    print(f"Total instances in depsgraph: {total_inst}")
    print(f"Instances INSIDE camera frustum: {inside_count}")
    print(f"Instances OUTSIDE camera frustum: {outside_count}")
