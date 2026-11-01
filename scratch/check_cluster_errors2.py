import struct

nc2_path = r"C:\blender-git\tessera_cenas_teste\Desert Rock2_V1.nc2"
with open(nc2_path, "rb") as f:
    hdr = f.read(88)
    clusterCount = struct.unpack("<I", hdr[8:12])[0]
    cluster_bytes = f.read(clusterCount * 64)

clusters = []
for i in range(clusterCount):
    chunk = cluster_bytes[i*64 : (i+1)*64]
    sphere = struct.unpack("<ffff", chunk[0:16])
    parent_sphere = struct.unpack("<ffff", chunk[16:32])
    err = struct.unpack("<f", chunk[32:36])[0]
    parent_err = struct.unpack("<f", chunk[36:40])[0]
    first_idx, idx_count, parent_grp, group, level = struct.unpack("<IIiII", chunk[40:60])
    flags, mat_idx = struct.unpack("<HH", chunk[60:64])
    clusters.append((i, level, err, parent_err, idx_count, sphere[3]))

by_level = {}
for c in clusters:
    by_level.setdefault(c[1], []).append(c)

print(f"Total levels: {len(by_level)}, total clusters: {len(clusters)}")
for lvl in sorted(by_level.keys()):
    items = by_level[lvl]
    avg_err = sum(x[2] for x in items) / len(items)
    avg_perr = sum(x[3] for x in items) / len(items)
    tot_tris = sum(x[4] for x in items) // 3
    print(f"Level {lvl:02d}: {len(items):3d} clusters, {tot_tris:5d} tris, avg_err={avg_err:.6f}, avg_parent_err={avg_perr:.6f}")

