import bpy

print("=== CHECKING OBJECTS AND GEOMETRY NODES ===")
for ob in bpy.data.objects:
    for m in ob.modifiers:
        if m.type == 'NODES' and m.node_group:
            print(f"Object '{ob.name}', Modifier '{m.name}', Group '{m.node_group.name}'")
            for node in m.node_group.nodes:
                name_lower = node.name.lower()
                tree_name = ""
                if hasattr(node, 'node_tree') and node.node_tree:
                    tree_name = node.node_tree.name.lower()
                if "cull" in name_lower or "cull" in tree_name or "camera" in name_lower or "camera" in tree_name:
                    print(f"   Node: '{node.name}' (type: {node.type}, tree: '{tree_name}')")
                    if hasattr(node, 'node_tree') and node.node_tree:
                        for subnode in node.node_tree.nodes:
                            print(f"      Subnode: '{subnode.name}' ({subnode.type})")
                            for inp in subnode.inputs:
                                if inp.is_linked:
                                    for link in inp.links:
                                        print(f"         Input '{inp.name}' linked from '{link.from_node.name}'.'{link.from_socket.name}'")

