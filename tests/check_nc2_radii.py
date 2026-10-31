import struct
import glob
import math

for f in glob.glob("C:/blender-git/tessera_cenas_teste/*.nc2"):
    with open(f, "rb") as fp:
        header_bytes = fp.read(88)
        bmin = struct.unpack("<3f", header_bytes[48:60])
        bmax = struct.unpack("<3f", header_bytes[60:72])
        diag = math.sqrt((bmax[0]-bmin[0])**2 + (bmax[1]-bmin[1])**2 + (bmax[2]-bmin[2])**2)
        radius = diag * 0.5
        name = f.split("\\")[-1]
        print(f"{name:<35} : radius={radius:.4f}m (bmin={bmin[0]:.2f}, bmax={bmax[0]:.2f})")

