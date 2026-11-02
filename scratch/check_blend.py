import bpy

print('=== SCENE OBJECTS ===')
for ob in bpy.data.objects:
    t_en = ob.get('tessera_enabled')
    nc2 = ob.get('tessera_nc2')
    mods = [m.type for m in ob.modifiers]
    if t_en is not None or nc2 is not None or len(mods) > 0:
        print(f'{ob.name}: type={ob.type}, t_en={t_en}, nc2={nc2}, mods={mods}')

print('\n=== SCENE PROPS ===')
for k in bpy.context.scene.keys():
    print(f'{k}: {bpy.context.scene[k]}')

