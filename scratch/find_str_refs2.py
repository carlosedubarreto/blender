import struct

exe_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe"
with open(exe_path, 'rb') as f:
    data = f.read()

pe_offset = struct.unpack('<I', data[0x3C:0x40])[0]
num_sections = struct.unpack('<H', data[pe_offset+6:pe_offset+8])[0]
opt_hdr_size = struct.unpack('<H', data[pe_offset+20:pe_offset+22])[0]
image_base = struct.unpack('<Q', data[pe_offset+24+24:pe_offset+24+32])[0]
section_table_offset = pe_offset + 24 + opt_hdr_size

sections = []
for i in range(num_sections):
    sec = data[section_table_offset + i*40: section_table_offset + (i+1)*40]
    name = sec[:8].rstrip(b'\x00').decode('latin1')
    vsize, vaddr, raw_size, raw_ptr = struct.unpack('<IIII', sec[8:24])
    sections.append((name, vaddr, vsize, raw_ptr, raw_size))

def offset_to_rva(offset):
    for name, vaddr, vsize, raw_ptr, raw_size in sections:
        if raw_ptr <= offset < raw_ptr + raw_size:
            return vaddr + (offset - raw_ptr)
    return None

str_offset = 70082384 # ScatterFlow Nano: %.2f of %.1f million...
str_rva = offset_to_rva(str_offset)
print(f"String file offset: {str_offset}, RVA: 0x{str_rva:X}, VA: 0x{image_base + str_rva:X}")

va_bytes = struct.pack('<Q', image_base + str_rva)
pos = 0
rdata_refs = []
while True:
    idx = data.find(va_bytes, pos)
    if idx == -1:
        break
    rdata_refs.append(idx)
    pos = idx + 8
print(f"Absolute VA refs: {len(rdata_refs)} at {[hex(r) for r in rdata_refs]}")

text_sec = next(s for s in sections if s[0] == '.text')
print(f"Scanning .text ({text_sec[4]} bytes) for RIP-relative refs...")
text_start = text_sec[3]
text_end = text_start + text_sec[4]
text_vaddr = text_sec[1]

# To find RIP-relative:
# An instruction like: lea reg, [rip + disp32] (e.g. 48 8d 15 disp32)
# At byte offset P in the file (RVA = text_vaddr + (P - text_start)):
# If the instruction ends at RVA_end = text_vaddr + (P - text_start) + 4
# disp32 = str_rva - RVA_end
# So disp32 changes for every byte! But notice:
# str_rva - (text_vaddr + (P - text_start) + 4) = disp32
# That means: disp32 + P = str_rva - text_vaddr + text_start - 4 = CONSTANT!
# CONSTANT C = str_rva - text_vaddr + text_start - 4
# So for every candidate, unpack 4 bytes as signed int: if val + P == C, then it matches!
C = str_rva - text_vaddr + text_start - 4
print(f"Magic constant C: 0x{C:X}")

matches = []
# Fast check in python
chunk = data[text_start:text_end]
for i in range(0, len(chunk) - 4, 1):
    val = struct.unpack('<i', chunk[i:i+4])[0]
    if val + (text_start + i) == C:
        matches.append(text_start + i)

print(f"Found {len(matches)} RIP-relative references in .text: {[hex(m) for m in matches]}")
for m in matches:
    # print preceding 16 bytes and following 16 bytes
    sub = data[m-16:m+20]
    print(f"At {hex(m)}: {sub.hex(' ')}")

