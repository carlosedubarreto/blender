import bpy

for name in ['Rocks_Small', 'Rocks_Large', 'SweetViolet', 'DewBerry', 'StickyWilly', 'StingingNettle', 'Terrain']:
    ob = bpy.data.objects.get(name)
    if ob:
        print(f"--- Object {name} ---")
        print("type:", ob.type, "modifiers:", [m.type for m in ob.modifiers])
        print("keys:", list(ob.keys()))
        for k in ob.keys():
            print(f"  {k} = {ob[k]}")
    else:
        print(f"Object {name} NOT FOUND")

