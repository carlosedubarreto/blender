import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

for ob in bpy.context.scene.objects:
    if 'tessera_nc2' in ob or 'tessera_enabled' in ob:
        ob['tessera_enabled'] = 1

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
region = next(r for r in area.regions if r.type == 'WINDOW')

ctx = {
    'window': win,
    'screen': win.screen,
    'area': area,
    'region': region,
    'scene': bpy.context.scene,
    'space_data': space,
}

with bpy.context.temp_override(**ctx):
    region.tag_redraw()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

depsgraph = bpy.context.evaluated_depsgraph_get()

print("\n" + "="*80)
print("TESSERA TELEMETRY DUMP:")
print("="*80)

total_triangles = 0
total_clusters = 0
total_active_inst = 0

for ob in bpy.context.scene.objects:
    eval_ob = ob.evaluated_get(depsgraph) if depsgraph else ob
    # Check both orig and eval
    props = {k: eval_ob[k] for k in eval_ob.keys() if 'tessera_stat_' in k}
    if not props:
        props = {k: ob[k] for k in ob.keys() if 'tessera_stat_' in k}
    if props:
        tris = props.get('tessera_stat_triangles', 0)
        clusters = props.get('tessera_stat_clusters', 0)
        tot_inst = props.get('tessera_stat_total_instances', 0)
        act_inst = props.get('tessera_stat_active_instances', 0)
        dist = props.get('tessera_stat_distance', 0)
        px_err = props.get('tessera_stat_pixel_error', 0)
        total_triangles += tris * max(1, act_inst)
        total_clusters += clusters
        total_active_inst += act_inst
        print(f"Object '{ob.name}':")
        print(f"  triangles/inst: {tris}, clusters: {clusters}, instances: {act_inst}/{tot_inst}, dist: {dist:.1f}m, err: {px_err}")

print(f"\nTOTAL active instances across scene: {total_active_inst}")
print(f"ESTIMATED total triangles drawn: {total_triangles:,}")

bpy.ops.wm.quit_blender()

