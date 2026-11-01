import struct
import math

nc2_path = r"C:\blender-git\tessera_cenas_teste\Rocks_Small.nc2"
with open(nc2_path, "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<I", f.read(4))[0]
    num_verts = struct.unpack("<I", f.read(4))[0]
    num_indices = struct.unpack("<I", f.read(4))[0]
    num_clusters = struct.unpack("<I", f.read(4))[0]
    print(f"NC2: magic={magic}, version={version}, verts={num_verts}, indices={num_indices}, clusters={num_clusters}")

    # Read positions: 3 floats each
    positions = []
    min_xyz = [1e9, 1e9, 1e9]
    max_xyz = [-1e9, -1e9, -1e9]
    for _ in range(num_verts):
        x, y, z = struct.unpack("<fff", f.read(12))
        min_xyz = [min(min_xyz[i], v) for i, v in enumerate([x, y, z])]
        max_xyz = [max(max_xyz[i], v) for i, v in enumerate([x, y, z])]

    center = [(min_xyz[i] + max_xyz[i]) / 2.0 for i in range(3)]
    radius = math.sqrt(sum((max_xyz[i] - min_xyz[i])**2 for i in range(3))) / 2.0
    print(f"Min: {min_xyz}, Max: {max_xyz}")
    print(f"Center: {center}, Radius: {radius:.4f}")

