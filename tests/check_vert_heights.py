import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()

g1 = bpy.data.objects.get("Ground_001")
g5 = bpy.data.objects.get("Ground_005")

# Test Rocks_Small on Ground_001
rs_ob = bpy.data.objects.get("Rocks_Small")
print("=== TESTING ROCKS_SMALL ===")
count = 0
for inst in depsgraph.object_instances:
    if inst.is_instance and inst.parent and inst.parent.name == "Rocks_Small":
        mat = inst.matrix_world.copy()
        pos = mat.translation
        # Raycast onto Ground_001 from above (pos.x, pos.y, 10.0) downwards
        ray_origin = mathutils.Vector((pos.x, pos.y, 5.0))
        ray_dir = mathutils.Vector((0, 0, -1))
        # Raycast in local space of Ground_001
        inv_g = g1.matrix_world.inverted()
        local_orig = inv_g @ ray_origin
        local_dir = (inv_g.to_3x3() @ ray_dir).normalized()
        success, hit_loc, hit_norm, hit_idx = g1.ray_cast(local_orig, local_dir)
        if success:
            world_hit = g1.matrix_world @ hit_loc
            # Check archetype vertices in world space
            arch_ob = inst.instance_object
            verts_z = []
            if arch_ob and arch_ob.data:
                for v in arch_ob.data.vertices[:20]:
                    wv = mat @ v.co
                    verts_z.append(wv.z)
            avg_vz = sum(verts_z)/len(verts_z) if verts_z else pos.z
            min_vz = min(verts_z) if verts_z else pos.z
            max_vz = max(verts_z) if verts_z else pos.z
            print(f"Inst at ({pos.x:.2f}, {pos.y:.2f}, Z_pos={pos.z:.3f}): Ground Z = {world_hit.z:.3f}, diff = {pos.z - world_hit.z:.3f}")
            print(f"   Archetype '{arch_ob.name}': vert Z range = [{min_vz:.3f}, {max_vz:.3f}] (diff to ground: min {min_vz - world_hit.z:.3f}, max {max_vz - world_hit.z:.3f})")
            count += 1
            if count >= 5:
                break

