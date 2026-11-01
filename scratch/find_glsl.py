import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

glsl_matches = set(re.findall(b"([a-zA-Z0-9_]+\\.glsl)", data))
print(f"GLSL files in PDB ({len(glsl_matches)}):")
for g in sorted(glsl_matches):
    if any(k in g.lower() for k in [b"nano", b"cluster", b"lod", b"scatter", b"cull"]):
        print(" ", g.decode("latin1", errors="ignore"))

