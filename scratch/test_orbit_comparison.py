import bpy
import time
import math
import mathutils

def get_gl_finish():
    try:
        import gpu
        if hasattr(gpu.state, 'finish'):
            return gpu.state.finish
    except Exception:
        pass
    import ctypes
    try:
        opengl32 = ctypes.windll.opengl32
        opengl32.glFinish.argtypes = []
        opengl32.glFinish.restype = None
        return opengl32.glFinish
    except Exception:
        return lambda: None

gl_finish = get_gl_finish()

def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return window, area, region, space, space.region_3d
    return None, None, None, None, None

window, area, region, space, rv3d = find_view3d()
space.shading.type = 'SOLID'
rv3d.view_perspective = 'PERSP'

for ob in bpy.data.objects:
    if ob.type == 'MESH':
        ob["tessera_enabled"] = 1
        ob["tessera_visualize_clusters"] = 0
        ob["tessera_pixel_error"] = 1.0
        ob["tessera_culling_margin"] = 0.02
        if hasattr(ob, "tessera"):
            ob.tessera.enabled = True

bpy.context.view_layer.update()

override = {
    "window": window,
    "screen": window.screen,
    "area": area,
    "region": region,
    "space_data": space,
    "region_data": rv3d,
}

# 5 warmup frames
for i in range(5):
    with bpy.context.temp_override(**override):
        bpy.ops.wm.redraw_timer(type='DRAW_SWAP', iterations=1)
gl_finish()

# Measure 10 frames with CPU and GPU breakdown
cpu_times = []
gpu_times = []
for i in range(10):
    yaw = (i / 10.0) * 2.0 * math.pi
    rv3d.view_rotation = mathutils.Euler((math.radians(60.0), 0.0, yaw), 'XYZ').to_quaternion()
    t0 = time.perf_counter()
    with bpy.context.temp_override(**override):
        bpy.ops.wm.redraw_timer(type='DRAW_SWAP', iterations=1)
    t1 = time.perf_counter()
    gl_finish()
    t2 = time.perf_counter()
    cpu_ms = (t1 - t0) * 1000.0
    gpu_ms = (t2 - t1) * 1000.0
    cpu_times.append(cpu_ms)
    gpu_times.append(gpu_ms)
    print(f"Frame {i:2d}: CPU={cpu_ms:6.1f} ms | GPU wait={gpu_ms:6.1f} ms | Total={cpu_ms+gpu_ms:6.1f} ms")

avg_cpu = sum(cpu_times) / len(cpu_times)
avg_gpu = sum(gpu_times) / len(gpu_times)
total = avg_cpu + avg_gpu
fps = 1000.0 / total
print(f"=== BREAKDOWN: CPU={avg_cpu:.1f}ms, GPU={avg_gpu:.1f}ms, Total={total:.1f}ms ({fps:.2f} FPS) ===")
bpy.ops.wm.quit_blender()
