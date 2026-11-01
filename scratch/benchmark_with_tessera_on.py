import bpy
import time
import mathutils

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
print(f"Loading {blend_path} ...")
bpy.ops.wm.open_mainfile(filepath=blend_path)

# Enable Tessera globally and on all objects
bpy.context.scene['tessera_enabled'] = 1
enabled_count = 0
for ob in bpy.context.scene.objects:
    ob['tessera_enabled'] = 1
    enabled_count += 1
print(f"Enabled Tessera on {enabled_count} objects!")

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
region = next(r for r in area.regions if r.type == 'WINDOW')
r3d = space.region_3d

ctx = {
    'window': win,
    'screen': win.screen,
    'area': area,
    'region': region,
    'scene': bpy.context.scene,
    'space_data': space,
}

# Warmup frames
print("\n--- WARMUP (5 frames) ---")
for i in range(5):
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

print("\n--- MEASURING 20 STATIONARY FRAMES ---")
stationary_times = []
for i in range(20):
    t0 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1000
    stationary_times.append(dt_ms)
    print(f"Stationary Frame {i:02d}: {dt_ms:.2f} ms ({1000/dt_ms:.1f} FPS)")

avg_stat = sum(stationary_times) / len(stationary_times)

print("\n--- MEASURING 20 ORBIT FRAMES (continuous camera rotation) ---")
orbit_times = []
rot_step = mathutils.Quaternion((0.0, 0.0, 1.0), 0.017)
for i in range(20):
    r3d.view_rotation = rot_step @ r3d.view_rotation
    t0 = time.perf_counter()
    with bpy.context.temp_override(**ctx):
        region.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1000
    orbit_times.append(dt_ms)
    print(f"Orbit Frame {i:02d}: {dt_ms:.2f} ms ({1000/dt_ms:.1f} FPS)")

avg_orbit = sum(orbit_times) / len(orbit_times)

print("\n=== SUMMARY WITH TESSERA ENABLED ===")
print(f"Stationary: {avg_stat:.2f} ms -> {1000/avg_stat:.1f} FPS")
print(f"Orbit:      {avg_orbit:.2f} ms -> {1000/avg_orbit:.1f} FPS")

bpy.ops.wm.quit_blender()
