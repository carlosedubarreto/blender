import struct

with open("C:/blender-git/tessera_cenas_teste/GalettaGrass_V3.nc2", "rb") as fp:
    header = fp.read(64)
    # Read first 10 clusters and last 10 clusters
    # Let's inspect TesseraCluster struct in C++

