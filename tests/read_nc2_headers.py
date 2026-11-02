import struct
import os

def read_nc2_header(path):
    with open(path, "rb") as f:
        data = f.read(88)
        u = struct.unpack("<12I 3f 3f 2f 2f", data)
        magic = u[0]
        version = u[1]
        clusterCount = u[2]
        levelCount = u[3]
        vertexCount = u[4]
        indexCount = u[5]
        flags = u[10]
        bmin = u[12:15]
        bmax = u[15:18]
        print(f"\nNC2 File: {os.path.basename(path)}")
        print(f"  Clusters: {clusterCount}, Verts: {vertexCount}, Indices: {indexCount}, Levels: {levelCount}")
        print(f"  BoundsMin: ({bmin[0]:.4f}, {bmin[1]:.4f}, {bmin[2]:.4f})")
        print(f"  BoundsMax: ({bmax[0]:.4f}, {bmax[1]:.4f}, {bmax[2]:.4f})")

base_dir = r"C:\blender-git\tessera_cenas_teste"
for name in ["Desert Rock3_V1.nc2", "SweetViolet_V1.nc2", "DewBerry_V1.nc2", "GalettaGrass_V1.nc2"]:
    p = os.path.join(base_dir, name)
    if os.path.exists(p):
        read_nc2_header(p)
