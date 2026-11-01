import re

pdb_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb"
with open(pdb_path, 'rb') as f:
    data = f.read()

# Find string references to .cc or .c
pattern = re.compile(rb'[C-Z]:\\[^\x00\r\n\t\x22\x3c\x3e]+\.(?:cc|cpp|c|h|hh)', re.IGNORECASE)
matches = set(pattern.findall(data))
print(f"Total source paths found: {len(matches)}")
for m in sorted(matches):
    s = m.decode('ascii', errors='ignore')
    if any(k in s.lower() for k in ['scatter', 'nano', 'tessera', 'workbench', 'eevee', 'draw_instance']):
        print(s)

