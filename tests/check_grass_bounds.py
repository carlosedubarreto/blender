import struct

with open("C:/blender-git/tessera_cenas_teste/GalettaGrass_V3.nc2", "rb") as fp:
    header_bytes = fp.read(88)
    # boundsMin is float[3] at offset 48, boundsMax is float[3] at offset 60
    bmin = struct.unpack("<3f", header_bytes[48:60])
    bmax = struct.unpack("<3f", header_bytes[60:72])
    print("bmin:", bmin)
    print("bmax:", bmax)
    diag = ((bmax[0]-bmin[0])**2 + (bmax[1]-bmin[1])**2 + (bmax[2]-bmin[2])**2)**0.5
    print("diag:", diag, "radius:", diag * 0.5)

