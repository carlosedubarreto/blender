import bpy

print('=== ALL MESHES IN SCENE ===')
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        t_en = ob.get('tessera_enabled')
        nc2 = ob.get('tessera_nc2')
        v_count = len(ob.data.vertices) if ob.data else 0
        mat_names = [slot.material.name if slot.material else 'None' for slot in ob.material_slots]
        print(f"Ob: '{ob.name}', verts={v_count}, hide_vp={ob.hide_viewport}, hide_render={ob.hide_render}, t_en={t_en}, nc2={nc2 is not None}, mats={mat_names}")

