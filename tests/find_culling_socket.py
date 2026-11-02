import bpy

ob = bpy.data.objects.get("Debris_Leafs")
m = ob.modifiers.get("GN_SCS_MOD")
inputs = m.properties.inputs

for item in m.node_group.interface.items_tree:
    if "cull" in item.name.lower() or "camera" in item.name.lower():
        val = getattr(inputs, item.identifier, None)
        print(f"Socket '{item.name}' (id: '{item.identifier}') = {val}")

