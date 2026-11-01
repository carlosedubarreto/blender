import re

pdb_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb"
with open(pdb_path, 'rb') as f:
    data = f.read()

idx = data.find(b'ScatterFlow Nano: preparing')
print("Found at:", idx)
if idx != -1:
    chunk = data[max(0, idx - 1000): min(len(data), idx + 2000)]
    strings = re.findall(rb'[\x20-\x7e]{4,}', chunk)
    for s in strings:
        print("  ", s.decode('ascii', errors='ignore'))
