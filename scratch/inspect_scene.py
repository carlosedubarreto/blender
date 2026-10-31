import bpy

print("=== SCENE INFO ===")
print("Render Engine:", bpy.context.scene.render.engine)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    print(f"Screen '{screen.name}' View3D Shading:", space.shading.type, space.shading.color_type)

tess_count = 0
for ob in bpy.data.objects:
    if ob.get("tessera_enabled", 0):
        tess_count += 1
print(f"Objects with tessera_enabled=1: {tess_count} / {len(bpy.data.objects)}")

