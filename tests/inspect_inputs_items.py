import bpy

ob = bpy.data.objects.get("Debris_Leafs")
m = ob.modifiers.get("GN_SCS_MOD")
inputs = m.properties.inputs
print("dir(inputs):", [a for a in dir(inputs) if not a.startswith("__")])
for a in dir(inputs):
    if not a.startswith("__") and not callable(getattr(inputs, a)):
        print(f"  {a} = {getattr(inputs, a)}")
