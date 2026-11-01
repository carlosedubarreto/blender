import bpy

p = bpy.types.View3DShading.bl_rna.properties.get("scatterflow_nano_debug")
if p:
    print("Debug enum items:")
    for item in p.enum_items:
        print(f"  {item.identifier}: {item.name} - {item.description}")

bpy.ops.wm.quit_blender()

