import bpy
import time

ob_rocks = bpy.data.objects.get("Rocks_Small")
print("Rocks_Small tessera_enabled:", ob_rocks.get("tessera_enabled"))
print("Rocks_Small tessera_nc2:", ob_rocks.get("tessera_nc2"))

for i in range(15):
    t0 = time.perf_counter()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    dt = (time.perf_counter() - t0) * 1000.0
    drawn = ob_rocks.get("tessera_stat_drawn_instances", -1) if ob_rocks else -1
    disp_ms = ob_rocks.get("tessera_stat_dispatch_ms", -1) if ob_rocks else -1
    gpu_ms = ob_rocks.get("tessera_stat_gpu_total_ms", -1) if ob_rocks else -1
    print(f"Frame {i:2d}: {dt:6.1f} ms | drawn_inst={drawn} disp={disp_ms} gpu={gpu_ms}")

