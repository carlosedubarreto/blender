import bpy

for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob['tessera_enabled'] = 1
        if hasattr(ob, 'tessera'):
            ob.tessera.enabled = True

for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.shading.type = 'SOLID'
            rv3d = space.region_3d
            rv3d.view_perspective = 'CAMERA'
            override = {'window': window, 'screen': window.screen, 'area': area}
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type='DRAW_SWAP', iterations=5)
                bpy.ops.screen.screenshot(filepath='C:/blender-git/tessera_cenas_teste/test_camera_view_drawn.png')

print("\n=== VIEWPORT CAMERA VIEW STATS ===")
for ob in bpy.data.objects:
    if ob.get("tessera_nc2") or ob.get("tessera_nc2_path"):
        drawn = ob.get("tessera_stat_drawn_instances", -1)
        total = ob.get("tessera_stat_total_instances", -1)
        clusters = ob.get("tessera_stat_clusters", -1)
        if drawn != -1 or total != -1:
            print(f"Object '{ob.name}': drawn={drawn}, total={total}, clusters={clusters}")

bpy.ops.wm.quit_blender()

