import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

wm = bpy.context.window_manager
win = wm.windows[0]
area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active

print("Render engine:", bpy.context.scene.render.engine)
print("Shading type:", space.shading.type)
print("Shading color_type:", space.shading.color_type)
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        print(f"Object: {ob.name}, modifiers: {[m.type for m in ob.modifiers]}, tessera props: {[k for k in ob.keys() if 'tessera' in k.lower()]}")

bpy.ops.wm.quit_blender()

