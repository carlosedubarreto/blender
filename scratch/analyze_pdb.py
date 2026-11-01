import re

pdb_path = r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb"

print("Scanning PDB for source file paths and symbols...")
found_sources = set()
found_symbols = set()

# Read in chunks
chunk_size = 10 * 1024 * 1024 # 10MB
with open(pdb_path, "rb") as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        
        # Search for .cc / .cpp / .hh / .h file paths
        matches = re.findall(b"[a-zA-Z0-9_\\\\/\\.-]+\\.(?:cc|cpp|hh|h)", chunk)
        for m in matches:
            m_str = m.decode("latin1", errors="ignore")
            if any(k in m_str.lower() for k in ["nano", "scatter", "virtual", "cluster", "tessera", "lod"]):
                found_sources.add(m_str)
                
        # Search for scatterflow_nano or nano symbols
        sym_matches = re.findall(b"(?:scatterflow|nano_[a-zA-Z0-9_]+|virtual_geometry[a-zA-Z0-9_]*)", chunk, re.IGNORECASE)
        for sm in sym_matches:
            found_symbols.add(sm.decode("latin1", errors="ignore"))

print("\n--- Matching Source Files in PDB ---")
for s in sorted(found_sources):
    print(" ", s)

print("\n--- Matching Symbols / Names in PDB ---")
for sym in sorted(found_symbols)[:50]:
    print(" ", sym)


