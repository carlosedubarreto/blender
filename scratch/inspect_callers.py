with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

targets = [b"mesh_cluster_lod_sync", b"DRW_mesh_batch_cache_get_cluster_lod", b"DrawList::dispatch", b"DrawList::select", b"DrawList::add"]
for t in targets:
    pos = 0
    print(f"\n--- Searching for {t.decode()} ---")
    while True:
        pos = data.find(t, pos)
        if pos == -1:
            break
        snippet = data[max(0, pos-120):min(len(data), pos+180)]
        clean = "".join(chr(b) if 32 <= b < 127 else "." for b in snippet)
        print(f"[{pos}]: {clean}")
        pos += len(t)

