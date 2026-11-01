with open(r"D:\_Blender_lib\Blender_assets\ScatterFlow\ScatterFlow-Blender-5.2.2-windows-x64_02_eevee\blender.pdb", "rb") as f:
    data = f.read()

for target in [b"workbench_engine.cc", b"eevee_instance", b"nano_visualization", b"GN_SCS_MOD", b"scatterflow_nano"]:
    idx = 0
    print(f"\nSearching for {target.decode()}:")
    for _ in range(5):
        idx = data.find(target, idx)
        if idx == -1:
            break
        snippet = data[max(0, idx-80):min(len(data), idx+120)]
        clean = "".join(chr(b) if 32 <= b < 127 else "." for b in snippet)
        print(f"  [{idx}]: {clean}")
        idx += len(target)

