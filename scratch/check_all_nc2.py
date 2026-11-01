import struct
import glob

for p in glob.glob(r"C:\blender-git\tessera_cenas_teste\*.nc2")[:6]:
    with open(p, "rb") as f:
        hdr = f.read(88)
        c, l, v, idx = struct.unpack("<IIII", hdr[8:24])
        bmin = struct.unpack("<fff", hdr[48:60])
        bmax = struct.unpack("<fff", hdr[60:72])
        chunk = f.read(64)
        err = struct.unpack("<f", chunk[32:36])[0]
        perr = struct.unpack("<f", chunk[36:40])[0]
        print(f"{p.split('\\')[-1]}: c={c}, dims={[bmax[i]-bmin[i] for i in range(3)]}, c0_err={err:.2f}, c0_perr={perr:.2f}")

