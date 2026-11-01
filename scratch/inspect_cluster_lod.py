import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Search for any symbols containing ClusterLOD or cluster_lod
cl_symbols = set()
for m in re.finditer(b"([a-zA-Z0-9_:]*(?:cluster_lod|ClusterLOD)[a-zA-Z0-9_:]*)", data, re.IGNORECASE):
    cl_symbols.add(m.group(1).decode("latin1", errors="ignore"))

print(f"Cluster LOD symbols found ({len(cl_symbols)}):")
for s in sorted(cl_symbols):
    print(" ", s)

