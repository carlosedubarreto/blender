import bpy

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

with bpy.context.temp_override(**override):
    bpy.ops.wm.redraw_timer(type='DRAW_SWAP', iterations=1)

print("=== STATS AT USER VIEW (16m) ===")
total_tris = 0
total_clusters = 0
total_inst = 0
for ob in bpy.data.objects:
    if "tessera_stat_triangles" in ob:
        tris = ob["tessera_stat_triangles"]
        clusters = ob.get("tessera_stat_clusters", 0)
        inst = ob.get("tessera_stat_drawn_instances", 0)
        if tris > 0 or inst > 0:
            print(f"  {ob.name:<25}: tris={tris:8,d} clusters={clusters:6,d} instances={inst:8,d}")
            total_tris += tris
            total_clusters += clusters
            total_inst += inst

print(f"TOTAL: tris={total_tris:,} | clusters={total_clusters:,} | drawn_instances={total_inst:,}")
import sys
sys.exit(0)

