import bpy
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()
for ob in bpy.context.scene.objects:
    if ob.name in ["Galetta Grass", "Debris_Leafs", "Rocks_Small", "PricklyPearCactus"]:
        bb = [mathutils.Vector(b) for b in ob.bound_box]
        center = sum(bb, mathutils.Vector()) / 8.0
        rad = max((b - center).length for b in bb)
        print(f"Ob: {ob.name}, local rad: {rad:.4f}")
        for m in ob.modifiers:
            print(f"  mod: {m.name} ({m.type})")

