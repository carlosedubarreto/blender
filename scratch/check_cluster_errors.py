import struct

nc2_path = r"C:\blender-git\tessera_cenas_teste\Desert Rock2_V1.nc2"
with open(nc2_path, "rb") as f:
    hdr = f.read(88)
    clusterCount = struct.unpack("<I", hdr[8:12])[0]
    print("clusterCount:", clusterCount)
    
    # clusters are at offset 88: each is 128 bytes
    cluster_bytes = f.read(clusterCount * 128)
    
clusters = []
for i in range(clusterCount):
    chunk = cluster_bytes[i*128 : (i+1)*128]
    sphere = struct.unpack("<ffff", chunk[0:16])
    err = struct.unpack("<f", chunk[16:20])[0]
    parent_err = struct.unpack("<f", chunk[20:24])[0]
    parent_sphere = struct.unpack("<ffff", chunk[24:40])
    idx_start, idx_count, v_start, v_count = struct.unpack("<IIII", chunk[40:56])
    mat_idx = struct.unpack("<H", chunk[56:58])[0]
    lod_level = struct.unpack("<B", chunk[58:59])[0]
    clusters.append((i, lod_level, err, parent_err, idx_count, sphere[3]))

# Group by lod_level
by_level = {}
for c in clusters:
    by_level.setdefault(c[1], []).append(c)

print(f"Total levels: {len(by_level)}")
for lvl in sorted(by_level.keys()):
    items = by_level[lvl]
    avg_err = sum(x[2] for x in items) / len(items)
    avg_perr = sum(x[3] for x in items) / len(items)
    tot_tris = sum(x[4] for x in items) // 3
    print(f"Level {lvl:02d}: {len(items):3d} clusters, {tot_tris:5d} tris, avg_err={avg_err:.6f}, avg_parent_err={avg_perr:.6f}")

