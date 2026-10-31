import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()
for ob in bpy.context.scene.objects:
    if ob.name in ["Galetta Grass", "Debris_Leafs", "Rocks_Large", "Ground_005"]:
        cam_pos = mathutils.Vector((5.78, 14.30 - 32.67, 1.15))
        dists = []
        for dup in depsgraph.object_instances:
            if dup.parent and dup.parent.original == ob:
                d = (dup.matrix_world.translation - cam_pos).length
                dists.append(d)
        if dists:
            dists.sort()
            print(f"Ob: {ob.name} ({len(dists)} insts):")
            for cutoff in [24, 26, 28, 30, 32, 34, 36]:
                cnt = sum(1 for d in dists if d <= cutoff)
                print(f"  <= {cutoff}m: {cnt}/{len(dists)} ({cnt/len(dists)*100:.1f}%)")

