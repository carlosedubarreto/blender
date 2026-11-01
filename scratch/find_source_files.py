import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Let's search for source paths: e.g. source/blender/draw or source\blender\draw
paths = set()
for match in re.finditer(b"([a-zA-Z]:\\\\[^\\x00\\r\\n]+\\.(?:cc|cpp))", data):
    p = match.group(1).decode("latin1", errors="ignore")
    if "source\\blender" in p.lower():
        paths.add(p)

print(f"Total source paths found: {len(paths)}")
# Filter for draw, eevee, workbench, geometry, mesh
draw_paths = [p for p in paths if any(k in p.lower() for k in ["draw", "eevee", "workbench", "geometry_nodes", "instances"])]
print(f"Draw-related source paths ({len(draw_paths)}):")
for p in sorted(draw_paths)[:60]:
    print(" ", p)

