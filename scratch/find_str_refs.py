import struct

exe_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe"
with open(exe_path, 'rb') as f:
    data = f.read()

# PE header
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

def rva_to_offset(rva):
    for name, vaddr, vsize, raw_ptr, raw_size in sections:
        if vaddr <= rva < vaddr + vsize:
            return raw_ptr + (rva - vaddr)
    return None

str_offset = 70082384 # ScatterFlow Nano: %.2f of %.1f million...
str_rva = offset_to_rva(str_offset)
print(f"String file offset: {str_offset}, RVA: 0x{str_rva:X}, VA: 0x{image_base + str_rva:X}")

# Search for references to str_rva (either absolute VA in .data/.rdata or rip-relative in .text)
va_bytes = struct.pack('<Q', image_base + str_rva)
rdata_refs = [m.start() for m in re.finditer(re.escape(va_bytes), data)]
print(f"Absolute VA refs: {len(rdata_refs)}")

# For RIP-relative (lea rdx, [rip + disp32]), in .text section:
text_sec = next(s for s in sections if s[0] == '.text')
text_data = data[text_sec[3]: text_sec[3] + text_sec[4]]

# Look for 32-bit displacements: disp32 = (target_rva - (instr_rva + instr_len))
# Let's search all occurrences of the displacement in .text
rip_refs = []
for i in range(len(text_data) - 4):
    disp = struct.unpack('<i', text_data[i:i+4])[0]
    instr_end_rva = text_sec[1] + i + 4
    if instr_end_rva + disp == str_rva:
        rip_refs.append(text_sec[3] + i)

print(f"RIP-relative refs in .text: {len(rip_refs)} at {[hex(r) for r in rip_refs]}")

