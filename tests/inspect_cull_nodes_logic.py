import bpy

tree = bpy.data.node_groups.get(".camera culling (No Camera)")
for n in tree.nodes:
    if n.name in ["Raycast.001", "Compare.004", "Switch.003", "Boolean Math.007", "Boolean Math.006", "Separate Geometry", "Switch", "Compare.001", "Compare.002", "Compare.003", "Switch.002"]:
        print(f"\nNode '{n.name}' ({n.type}):")
        if hasattr(n, 'operation'):
            print(f"   operation: {n.operation}")
        for inp in n.inputs:
            def_val = inp.default_value if hasattr(inp, 'default_value') else None
            links = [(l.from_node.name, l.from_socket.name) for l in inp.links]
            print(f"   input '{inp.name}': default={def_val}, links={links}")
        for outp in n.outputs:
            links = [(l.to_node.name, l.to_socket.name) for l in outp.links]
            print(f"   output '{outp.name}': links={links}")

