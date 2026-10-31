import bpy
import mathutils

scene = bpy.context.scene
mesh_obs = [ob for ob in scene.objects if ob.type == 'MESH']

# Calculate bounds and orbit distance as in benchmark
all_min = mathutils.Vector((1e9, 1e9, 1e9))
all_max = mathutils.Vector((-1e9, -1e9, -1e9))
for ob in mesh_obs:
    for b in ob.bound_box:
        w = ob.matrix_world @ mathutils.Vector(b)
        for i in range(3):
            all_min[i] = min(all_min[i], w[i])
            all_max[i] = max(all_max[i], w[i])
center = 0.5 * (all_min + all_max)
diag = (all_max - all_min).length
orbit_dist = max(diag * 0.75, 4.0)

print(f"Center: {center}, Diag: {diag:.2f}, Orbit dist: {orbit_dist:.2f}")

depsgraph = bpy.context.evaluated_depsgraph_get()
for ob in mesh_obs:
    if ob.name.startswith("OSGN_Proxy"):
        continue
    eval_ob = ob.evaluated_get(depsgraph)
    # Check duplis / instances
    eval_ob.update_tag()
    duplis = []
    if eval_ob.is_instancer:
        for dup in depsgraph.object_instances:
            if dup.parent and dup.parent.original == ob:
                duplis.append(dup.matrix_world.copy())
    
    print(f"Object: {ob.name:<25} | Mesh Verts: {len(ob.data.vertices):>7} | Instances: {len(duplis)}")

