import struct

with open("C:/blender-git/tessera_cenas_teste/GalettaGrass_V3.nc2", "rb") as fp:
    header_bytes = fp.read(88)
    magic, version, cluster_count, level_count = struct.unpack("<IIII", header_bytes[:16])
    print(f"cluster_count={cluster_count}, level_count={level_count}")
    
    levels = {}
    for i in range(cluster_count):
        data = fp.read(64)
        c_sphere = struct.unpack("<4f", data[0:16])
        p_sphere = struct.unpack("<4f", data[16:32])
        err, p_err, first_idx, idx_cnt, p_grp, grp, lvl, flags, mat = struct.unpack("<ffIIiIIHH", data[32:64])
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append((err, p_err, idx_cnt))

for lvl in sorted(levels.keys()):
    errs = [x[0] for x in levels[lvl]]
    p_errs = [x[1] for x in levels[lvl] if x[1] < 1e30]
    idx_cnts = [x[2] for x in levels[lvl]]
    avg_err = sum(errs)/len(errs)
    avg_perr = sum(p_errs)/len(p_errs) if p_errs else 1e38
    print(f"Level {lvl:2d}: count={len(errs):5d}, avg_idx={sum(idx_cnts)/len(idx_cnts):.0f}, avg_err={avg_err:.6f}, avg_parent_err={avg_perr:.6f}")
