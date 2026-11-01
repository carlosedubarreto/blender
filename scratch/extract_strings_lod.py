import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe", "rb") as f:
    data = f.read()

# Search for strings containing "cluster" or "nano" or "lod"
str_matches = set()
for m in re.finditer(b"([a-zA-Z0-9_ /\\.:\\-]{4,80})", data):
    s = m.group(1).decode("latin1", errors="ignore")
    s_low = s.lower()
    if any(k in s_low for k in ["cluster_lod", "scatterflow_nano", "gn_scs_mod", "drawlist"]):
        str_matches.add(s)

print(f"Strings found in blender.exe ({len(str_matches)}):")
for s in sorted(str_matches):
    print(" ", s)

