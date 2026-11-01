import bpy

wm_props = bpy.types.WindowManager.bl_rna.properties
for p in wm_props:
    if any(x in p.identifier.lower() for x in ["nano", "scatter", "cull", "virtual"]):
        print(f"WM property: {p.identifier} (type={p.type}, description={p.description})")

shading_props = bpy.types.View3DShading.bl_rna.properties
for p in shading_props:
    if any(x in p.identifier.lower() for x in ["nano", "scatter", "cull", "virtual"]):
        print(f"Shading property: {p.identifier} (type={p.type}, description={p.description})")

bpy.ops.wm.quit_blender()

