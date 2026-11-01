import re

exe_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe"
with open(exe_path, 'rb') as f:
    data = f.read()

words = ['scatterflow', 'nano', 'cull_size', 'GN_SCS_MOD']
for w in words:
    matches = [m.start() for m in re.finditer(w.encode('ascii'), data, re.IGNORECASE)]
    print(f"Word '{w}': {len(matches)} occurrences")
    for pos in matches[:5]:
        start = max(0, pos - 60)
        end = min(len(data), pos + 100)
        chunk = data[start:end]
        printable = "".join([chr(b) if 32 <= b <= 126 else "." for b in chunk])
        print(f"  [{pos}] {printable}")

