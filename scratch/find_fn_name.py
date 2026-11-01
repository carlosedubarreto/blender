import struct

exe_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.exe"
with open(exe_path, 'rb') as f:
    data = f.read()

offset = 0xf29c23
text_vaddr = 0x1000
text_start = 0x400
rva = text_vaddr + (offset - text_start)

# Let's see what symbols in PDB are closest to this RVA!
pdb_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb"
# We can search the PDB for functions with RVA close to rva (0xF2A823)
print(f"File offset: 0x{offset:X}, RVA: 0x{rva:X}")

