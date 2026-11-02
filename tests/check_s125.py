import bpy

ob = bpy.data.objects.get("Debris_Leafs")
m = ob.modifiers.get("GN_SCS_MOD")
inputs = m.properties.inputs
s125 = getattr(inputs, 'Socket_125')
print("dir(Socket_125):", [a for a in dir(s125) if not a.startswith("__")])
for a in dir(s125):
    if not a.startswith("__"):
        print(f"  {a} = {getattr(s125, a)}")

