import re
import struct

pdb_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb"
with open(pdb_path, 'rb') as f:
    data = f.read()

target_rva = 0xF2A823

# In PDB DBI/GSSI streams, S_PUB32 records have:
# 2 bytes record len, 2 bytes type (0x110E for S_PUB32), 4 bytes pubflags, 4 bytes offset, 2 bytes section, null-terminated name
# Let's search for 0x110E
pub_syms = []
pos = 0
while True:
    idx = data.find(b'\x0e\x11', pos)
    if idx == -1:
        break
    if idx >= 2:
        rlen = struct.unpack('<H', data[idx-2:idx])[0]
        if 14 <= rlen <= 500:
            flags, sec_offset, sec_num = struct.unpack('<IIH', data[idx+2:idx+12])
            if sec_num == 1: # .text is section 1
                name_end = data.find(b'\x00', idx+12)
                if name_end != -1 and name_end <= idx - 2 + rlen + 2:
                    name = data[idx+12:name_end].decode('ascii', errors='ignore')
                    if name:
                        pub_syms.append((sec_offset, name))
    pos = idx + 2

pub_syms.sort()
print(f"Total public symbols in text: {len(pub_syms)}")
# Find closest symbol before target_rva
closest = None
for offset, name in pub_syms:
    if offset <= target_rva:
        closest = (offset, name)
    else:
        break

print(f"Target RVA: 0x{target_rva:X}")
if closest:
    print(f"Function: {closest[1]} (starts at 0x{closest[0]:X}, diff = 0x{target_rva - closest[0]:X})")

