import bpy

ob = bpy.data.objects.get("Debris_Leafs")
m = ob.modifiers.get("GN_SCS_MOD")

print("dir(m):", [attr for attr in dir(m) if not attr.startswith("__")])

# Try to find input sockets and their values
tree = m.node_group
for item in tree.interface.items_tree:
    if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
        try:
            val = m[item.identifier]
            print(f"Socket '{item.name}' (id: '{item.identifier}') = {val}")
        except Exception as e:
            # Maybe accessed by item.name?
            try:
                val = m[item.name]
                print(f"Socket '{item.name}' = {val}")
            except Exception as e2:
                pass

