import bpy

tree = bpy.data.node_groups.get(".camera culling (no camera)")
if not tree:
    for g in bpy.data.node_groups:
        if "camera culling" in g.name.lower():
            tree = g
            break

print("TREE:", tree.name if tree else "None")
if tree:
    print("\n--- NODES ---")
    for n in tree.nodes:
        print(f"Node: {n.name} ({n.type})")
        if n.type == 'SEPARATE_GEOMETRY':
            print(f"   Separate Geometry: {n.name}")
        if n.type == 'BOOLEAN_MATH':
            print(f"   Boolean Math operation: {n.operation}")
        if n.type == 'COMPARE':
            print(f"   Compare operation: {n.operation}, data_type: {n.data_type}")

    print("\n--- LINKS ---")
    for l in tree.links:
        print(f"{l.from_node.name}.'{l.from_socket.name}' -> {l.to_node.name}.'{l.to_socket.name}'")

