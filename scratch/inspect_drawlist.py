import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Search for any symbols in blender::draw::cluster_lod
symbols = set()
for m in re.finditer(b"blender::draw::cluster_lod::([a-zA-Z0-9_:]+)", data):
    sym = m.group(1).decode("latin1", errors="ignore")
    symbols.add(sym)

print(f"Total cluster_lod symbols ({len(symbols)}):")
for s in sorted(symbols):
    print(" ", s)

# Search for eevee or workbench functions that reference cluster_lod
draw_refs = set()
for m in re.finditer(b"([a-zA-Z0-9_:]*(?:cluster_lod|ClusterLod)[a-zA-Z0-9_:]*)", data):
    sym = m.group(1).decode("latin1", errors="ignore")
    if any(k in sym.lower() for k in ["eevee", "workbench", "draw", "engine", "sync"]):
        draw_refs.add(sym)

print(f"\nDraw engine references ({len(draw_refs)}):")
for s in sorted(draw_refs):
    print(" ", s)

