import struct

for f in ["C:/blender-git/tessera_cenas_teste/DeciduousDebris_LeafScatter.nc2",
          "C:/blender-git/tessera_cenas_teste/GalettaGrass_V1.nc2",
          "C:/blender-git/tessera_cenas_teste/GalettaGrass_V2.nc2"]:
    print("=" * 60)
    print("Inspecting:", f)
    with open(f, "rb") as fp:
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
            idx_cnts = [x[2] for x in levels[lvl]]
            print(f"  Level {lvl:2d}: count={len(errs):5d}, avg_idx={sum(idx_cnts)/len(idx_cnts):.0f}")

