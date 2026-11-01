import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

print("\n--- SCENE PROPERTIES ---")
for k, v in bpy.context.scene.items():
    print(f"scene['{k}'] = {v}")

print("\n--- WINDOW MANAGER PROPERTIES ---")
for k, v in bpy.context.window_manager.items():
    print(f"wm['{k}'] = {v}")

print("\n--- VIEW3D SPACE PROPERTIES ---")
area = next(a for a in bpy.context.window_manager.windows[0].screen.areas if a.type == 'VIEW_3D')
space = area.spaces.active
for k, v in space.items():
    print(f"space['{k}'] = {v}")

# Check all attributes of space
for attr in dir(space):
    if 'scatter' in attr.lower() or 'nano' in attr.lower() or 'lod' in attr.lower() or 'anno' in attr.lower():
        print(f"space.{attr} = {getattr(space, attr, None)}")

for attr in dir(bpy.context.scene):
    if 'scatter' in attr.lower() or 'nano' in attr.lower() or 'lod' in attr.lower() or 'anno' in attr.lower():
        print(f"scene.{attr} = {getattr(bpy.context.scene, attr, None)}")

for ob in bpy.context.scene.objects:
    for k, v in ob.items():
        if 'scatter' in k.lower() or 'nano' in k.lower() or 'anno' in k.lower():
            print(f"ob['{ob.name}']['{k}'] = {v}")

bpy.ops.wm.quit_blender()

