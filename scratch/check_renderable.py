import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

print("--- RENDERABLE OBJECTS ---")
for ob in bpy.context.view_layer.objects:
    if ob.type == 'MESH':
        has_nc2 = ob.get("tessera_nc2") or ob.get("tessera_enabled")
        mods = [m.name for m in ob.modifiers]
        print(f"Name: {ob.name:30} | Visible: {str(ob.visible_get()):5} | Has Tessera: {str(bool(has_nc2)):5} | Mods: {mods}")

bpy.ops.wm.quit_blender()

