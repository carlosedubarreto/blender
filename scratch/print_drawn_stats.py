import bpy

window = bpy.context.window_manager.windows[0]
for area in window.screen.areas:
    if area.type == 'VIEW_3D':
        for region in area.regions:
            if region.type == 'WINDOW':
                with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                break
        break

print("\n=== DRAWN INSTANCES PER EMITTER ===")
total_drawn = 0
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        cnt = ob.get("tessera_stat_drawn_instances", 0)
        clusters = ob.get("tessera_stat_clusters", 0)
        tris = ob.get("tessera_stat_triangles", 0)
        if cnt > 0 or clusters > 0:
            print(f"  {ob.name:<25}: {cnt:>6} instances, {clusters:>5} clusters, {tris:>8} tris")
            total_drawn += cnt
print(f"Total drawn instances: {total_drawn}")
bpy.ops.wm.quit_blender()

