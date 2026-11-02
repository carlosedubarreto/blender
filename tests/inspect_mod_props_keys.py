import bpy

ob = bpy.data.objects.get("Debris_Leafs")
m = ob.modifiers.get("GN_SCS_MOD")
props = m.properties
print("type(props):", type(props))
for k in dir(props):
    if not k.startswith("__"):
        print(f"prop attr: {k} = {getattr(props, k)}")

