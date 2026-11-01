import struct

nc2_path = r"C:\blender-git\tessera_cenas_teste\Desert Rock2_V1.nc2"
with open(nc2_path, "rb") as f:
    hdr_bytes = f.read(88)
    bmin = struct.unpack("<fff", hdr_bytes[48:60])
    bmax = struct.unpack("<fff", hdr_bytes[60:72])
    print(f"bmin={bmin}, bmax={bmax}")
    print(f"dims: {[bmax[i] - bmin[i] for i in range(3)]}")

