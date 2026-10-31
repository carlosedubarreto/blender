import struct
import glob
import os

nc2_files = glob.glob("C:/blender-git/tessera_cenas_teste/*.nc2") + glob.glob("D:/_Code/Meu/Nanite/*.nc2")
for f in nc2_files:
    if "Galetta" in f:
        print("Found:", f)
        with open(f, "rb") as fp:
            magic = fp.read(4)
            version = struct.unpack("<I", fp.read(4))[0]
            num_verts = struct.unpack("<I", fp.read(4))[0]
            num_tris = struct.unpack("<I", fp.read(4))[0]
            num_clusters = struct.unpack("<I", fp.read(4))[0]
            print(f"  verts={num_verts}, tris={num_tris}, clusters={num_clusters}")

