import bpy
import mathutils

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

for ob in scene.objects:
    if "Galetta" in ob.name or "Grass" in ob.name or "Debris" in ob.name:
        eval_ob = ob.evaluated_get(depsgraph)
        print(f"Name: {ob.name}, type: {ob.type}")
        scales = []
        dists = []
        cam_pos = mathutils.Vector((5.78, 14.30 - 32.67, 1.15)) # approx orbit
        for dup in depsgraph.object_instances:
            if dup.parent and dup.parent.original == ob:
                mat = dup.matrix_world
                scale = mat.to_scale()
                scales.append(scale.length)
                dist = (mat.translation - cam_pos).length
                dists.append(dist)
        if scales:
            print(f"  Instances: {len(scales)}, Scale min: {min(scales):.4f}, max: {max(scales):.4f}, avg: {sum(scales)/len(scales):.4f}")
            print(f"  Dist min: {min(dists):.2f}, max: {max(dists):.2f}")

