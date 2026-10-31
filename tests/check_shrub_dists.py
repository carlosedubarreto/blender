import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()
cam_pos = mathutils.Vector((5.78, 14.30 - 32.67, 1.15))

for name in ["SweetViolet", "DewBerry", "StickyWilly", "StingingNettle"]:
    ob = bpy.data.objects.get(name)
    if not ob:
        continue
    dists = []
    for dup in depsgraph.object_instances:
        if dup.parent and dup.parent.original == ob:
            d = (dup.matrix_world.translation - cam_pos).length
            dists.append(d)
    if dists:
        dists.sort()
        print(f"Ob: {name} (total {len(dists)}):")
        for cutoff in [22, 24, 26, 28, 30, 32]:
            cnt = sum(1 for d in dists if d <= cutoff)
            print(f"  <= {cutoff}m: {cnt}/{len(dists)} ({cnt/len(dists)*100:.1f}%)")

