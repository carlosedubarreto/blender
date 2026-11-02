import bpy

tree = bpy.data.node_groups.get(".camera culling (No Camera)")
print("=== DOWNSTREAM FROM ACCUMULATE FIELD ===")
for name in ["Boolean Math.006", "Switch.001", "Boolean Math.007", "Capture Attribute", "Sample Index.001", "Separate Geometry", "Separate Geometry.001"]:
    n = tree.nodes.get(name)
    if n:
        print(f"\nNode '{n.name}' ({n.type}):")
        if hasattr(n, 'operation'):
            print(f"  operation: {n.operation}")
        for i, inp in enumerate(n.inputs):
            links = [(l.from_node.name, l.from_socket.name) for l in inp.links]
            print(f"  in[{i}] '{inp.name}': val={getattr(inp, 'default_value', None)}, links={links}")
        for i, outp in enumerate(n.outputs):
            links = [(l.to_node.name, l.to_socket.name) for l in outp.links]
            print(f"  out[{i}] '{outp.name}': links={links}")

