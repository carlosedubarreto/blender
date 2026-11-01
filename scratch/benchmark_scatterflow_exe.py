import bpy
import time
import sys

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
print("Loading:", blend_path)
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
shading = space.shading

print("\nInitial State:")
print("  wm.scatterflow_nano:", getattr(wm, "scatterflow_nano", None))
print("  wm.scatterflow_nano_modifiers:", getattr(wm, "scatterflow_nano_modifiers", None))
print("  shading.use_scatterflow_nano:", getattr(shading, "use_scatterflow_nano", None))
print("  shading.scatterflow_nano_error:", getattr(shading, "scatterflow_nano_error", None))
print("  shading.scatterflow_nano_cull_size:", getattr(shading, "scatterflow_nano_cull_size", None))
print("  shading.type:", shading.type)

# Function to benchmark redraw FPS
def benchmark_viewport(name, warmup=5, frames=30):
    region = next(r for r in area.regions if r.type == 'WINDOW')
    ctx = {
        'window': win,
        'screen': win.screen,
        'area': area,
        'region': region,
        'scene': bpy.context.scene,
        'space_data': space,
    }
    
    # Warmup
    for _ in range(warmup):
        with bpy.context.temp_override(**ctx):
            region.tag_redraw()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            
    # Timed loop
    t0 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        for _ in range(frames):
            region.tag_redraw()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    ms_per_frame = (total_time / frames) * 1000.0
    fps = frames / total_time
    print(f"[{name}] {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame, total {total_time:.3f}s for {frames} frames)")
    return fps, ms_per_frame

# Test 1: Nano OFF
if hasattr(wm, "scatterflow_nano"):
    wm.scatterflow_nano = False
if hasattr(shading, "use_scatterflow_nano"):
    shading.use_scatterflow_nano = False

fps_off, ms_off = benchmark_viewport("ScatterFlow Nano OFF")

# Test 2: Nano ON (Default error 1.0)
if hasattr(wm, "scatterflow_nano"):
    wm.scatterflow_nano = True
if hasattr(wm, "scatterflow_nano_modifiers"):
    wm.scatterflow_nano_modifiers = "GN_SCS_MOD"
if hasattr(shading, "use_scatterflow_nano"):
    shading.use_scatterflow_nano = True
if hasattr(shading, "scatterflow_nano_error"):
    shading.scatterflow_nano_error = 1.0

fps_on, ms_on = benchmark_viewport("ScatterFlow Nano ON (error=1.0)")

# Test 3: Nano ON with error 2.0 (coarser)
if hasattr(shading, "scatterflow_nano_error"):
    shading.scatterflow_nano_error = 2.0
fps_on2, ms_on2 = benchmark_viewport("ScatterFlow Nano ON (error=2.0)")

# Test 4: Nano ON with cull_size = 5.0 (thin out small objects)
if hasattr(shading, "scatterflow_nano_cull_size"):
    shading.scatterflow_nano_cull_size = 5.0
fps_cull, ms_cull = benchmark_viewport("ScatterFlow Nano ON (error=2.0, cull_size=5.0)")

print("\n" + "="*80)
print(f"SUMMARY COMPARISON in {sys.executable}:")
print(f"  Nano OFF : {fps_off:.1f} FPS ({ms_off:.2f} ms)")
print(f"  Nano ON  : {fps_on:.1f} FPS ({ms_on:.2f} ms) -> Speedup: {fps_on/max(fps_off, 0.001):.2f}x")
print(f"  Nano 2.0 : {fps_on2:.1f} FPS ({ms_on2:.2f} ms)")
print(f"  Nano Cull: {fps_cull:.1f} FPS ({ms_cull:.2f} ms)")
print("="*80)

bpy.ops.wm.quit_blender()

