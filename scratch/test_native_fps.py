import bpy
import time

for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'

bpy.context.view_layer.update()

times = []
for i in range(10):
    t0 = time.perf_counter()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    times.append((time.perf_counter() - t0) * 1000.0)

avg_ms = sum(times[2:]) / len(times[2:])
fps = 1000.0 / avg_ms if avg_ms > 0 else 0
print(f"=== NATIVE BLENDER 4.5.2 BENCHMARK: {avg_ms:.2f} ms ({fps:.2f} FPS) ===")
print("Frame times:", [f"{t:.1f}ms" for t in times])
import sys
sys.exit(0)

