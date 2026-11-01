import struct

nc2_path = r"C:\blender-git\tessera_cenas_teste\Rocks_Small.nc2"
with open(nc2_path, "rb") as f:
    hdr_bytes = f.read(88)
    magic, version, clusterCount, levelCount, vertexCount, indexCount = struct.unpack("<IIIIII", hdr_bytes[:24])
    bmin = struct.unpack("<fff", hdr_bytes[48:60])
    bmax = struct.unpack("<fff", hdr_bytes[60:72])
    print(f"clusterCount={clusterCount}, levelCount={levelCount}, vertexCount={vertexCount}, indexCount={indexCount}")
    print(f"bmin={bmin}, bmax={bmax}")

