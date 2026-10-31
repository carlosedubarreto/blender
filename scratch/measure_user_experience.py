import bpy
import time
import sys

print("="*60)
print("TEST WITH TESSERA ENABLED ON ALL OBJECTS")
print("="*60)

for ob in bpy.data.objects:
    if ob.type == 'MESH':
        ob["tessera_enabled"] = 1

bpy.context.view_layer.update()

times = []
for i in range(20):
    t0 = time.perf_counter()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)

avg_ms = sum(times[2:]) / len(times[2:])
fps = 1000.0 / avg_ms if avg_ms > 0 else 0
print(f"With Tessera Enabled: {avg_ms:.2f} ms per frame ({fps:.2f} FPS)")
print(f"First 3 frame times: {[f'{t:.1f}ms' for t in times[:3]]}")
print("="*60)
sys.exit(0)
