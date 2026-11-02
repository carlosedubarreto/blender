import bpy

depsgraph = bpy.context.evaluated_depsgraph_get()

# Let's inspect which objects have instances with Z coordinates or bounding boxes
for ob in bpy.data.objects:
    if ob.type == 'MESH':
        pts = []
        for inst in depsgraph.object_instances:
            if inst.is_instance and inst.parent and inst.parent.name == ob.name:
                pts.append(inst.matrix_world.translation.copy())
        if pts:
            avg_z = sum(p.z for p in pts) / len(pts)
            print(f"Object '{ob.name}': {len(pts)} insts, avg_z = {avg_z:.3f}, min_z = {min(p.z for p in pts):.3f}, max_z = {max(p.z for p in pts):.3f}")

