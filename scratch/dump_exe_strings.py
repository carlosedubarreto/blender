import re

exe_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe"
with open(exe_path, 'rb') as f:
    data = f.read()

def dump_strings_around(offset, radius=1000):
    start = max(0, offset - radius)
    end = min(len(data), offset + radius)
    chunk = data[start:end]
    strings = re.findall(rb'[\x20-\x7e]{4,}', chunk)
    print(f"=== STRINGS AROUND {offset} ===")
    for s in strings:
        print("  ", s.decode('ascii', errors='ignore'))

dump_strings_around(70082328, 1200)
dump_strings_around(69520324, 1200)
dump_strings_around(72336737, 1200)

