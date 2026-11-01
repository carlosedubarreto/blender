import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Let's search for any function or class symbol containing "Nano" or "nano" in blender::
matches = re.findall(b"blender::[a-zA-Z0-9_:]*nano[a-zA-Z0-9_:]*", data, re.IGNORECASE)
print(f"Unique blender::*nano* symbols ({len(set(matches))}):")
for m in sorted(set(matches)):
    print(" ", m.decode("latin1", errors="ignore"))

# Search for virtual_geometry symbols
vg_matches = re.findall(b"blender::[a-zA-Z0-9_:]*virtual[a-zA-Z0-9_:]*", data, re.IGNORECASE)
print(f"\nUnique blender::*virtual* symbols ({len(set(vg_matches))}):")
for m in sorted(set(vg_matches)):
    print(" ", m.decode("latin1", errors="ignore"))

# Search for any string literals with "scatterflow_nano"
str_matches = re.findall(b"[a-zA-Z0-9_]*scatterflow_nano[a-zA-Z0-9_]*", data, re.IGNORECASE)
print(f"\nUnique scatterflow_nano strings ({len(set(str_matches))}):")
for m in sorted(set(str_matches)):
    print(" ", m.decode("latin1", errors="ignore"))

