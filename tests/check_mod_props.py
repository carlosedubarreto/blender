import bpy

for ob in bpy.data.objects:
    for m in ob.modifiers:
        if m.type == 'NODES' and m.node_group:
            print(f"\nObject {ob.name}, Modifier {m.name}")
            for item in m.node_group.interface.items_tree:
                if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                    try:
                        val = m[item.identifier]
                        if "cull" in item.name.lower() or "cam" in item.name.lower() or "occlu" in item.name.lower():
                            print(f"   Input '{item.name}' ({item.identifier}) = {val}")
                    except Exception as e:
                        pass
