with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

# Search for "draw_cluster_lod.cc"
idx = 0
while True:
    idx = data.find(b"draw_cluster_lod", idx)
    if idx == -1:
        break
    snippet = data[max(0, idx-100):min(len(data), idx+200)]
    clean = "".join(chr(b) if 32 <= b < 127 else "." for b in snippet)
    print(f"[{idx}]: {clean}")
    idx += 16

