import bpy

print("=== CHECKING GN_ScatterSystem.005 ===")
for g in bpy.data.node_groups:
    if "scattersystem" in g.name.lower():
        print(f"\nNode Group: {g.name}")
        for n in g.nodes:
            if "cull" in n.name.lower() or "camera" in n.name.lower() or (n.type == 'GROUP' and n.node_tree and 'cull' in n.node_tree.name.lower()):
                print(f"  Node: {n.name} (type: {n.type})")
                if n.node_tree:
                    print(f"    Sub-tree: {n.node_tree.name}")
                for inp in n.inputs:
                    val = inp.default_value if hasattr(inp, 'default_value') else None
                    is_l = inp.is_linked
                    print(f"    Input '{inp.name}': default={val}, linked={is_l}")
                    if is_l:
                        for l in inp.links:
                            print(f"      <- {l.from_node.name}.'{l.from_socket.name}'")
                for outp in n.outputs:
                    print(f"    Output '{outp.name}': linked={outp.is_linked}")
                    if outp.is_linked:
                        for l in outp.links:
                            print(f"      -> {l.to_node.name}.'{l.to_socket.name}'")

