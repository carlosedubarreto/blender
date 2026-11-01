import re

with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Search for *.obj names in bf_draw
obj_matches = re.findall(b"bf_draw\\.dir\\\\Release\\\\([a-zA-Z0-9_]+\\.obj)", data)
print(f"Unique bf_draw .obj files ({len(set(obj_matches))}):")
for obj in sorted(set(obj_matches)):
    print(" ", obj.decode("latin1", errors="ignore"))

# Search for any new obj files that might be custom
all_objs = re.findall(b"\\\\([a-zA-Z0-9_]+\\.obj)", data)
custom_objs = [o for o in set(all_objs) if any(k in o.lower() for k in [b"nano", b"scatter", b"tessera", b"virtual", b"cluster", b"lod"])]
print(f"\nCustom .obj files across all libs ({len(custom_objs)}):")
for o in sorted(custom_objs):
    print(" ", o.decode("latin1", errors="ignore"))

