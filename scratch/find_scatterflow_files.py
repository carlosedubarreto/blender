import bpy
import sys

print("Addon path for bl_ext.user_default.scatterflow:")
addon = bpy.context.preferences.addons.get("bl_ext.user_default.scatterflow")
if addon:
    print("  Addon module:", addon.module)
    import bl_ext.user_default.scatterflow as sf
    print("  File:", getattr(sf, "__file__", "unknown"))

# Also search sys.path for scatterflow
for p in sys.path:
    if "scatter" in p.lower() or "nano" in p.lower():
        print("  Sys path:", p)

# Search RNA types for scatterflow_nano
print("\nSearching RNA Structs for 'nano' or 'scatterflow':")
for rna_type in dir(bpy.types):
    if "nano" in rna_type.lower() or "scatterflow" in rna_type.lower():
        print("  RNA Type:", rna_type)

# Inspect View3DShading properties
print("\nView3DShading properties:")
sh = bpy.context.window_manager.windows[0].screen.areas[0].spaces[0].shading if bpy.context.window_manager.windows else None
for p in dir(bpy.types.View3DShading):
    if "scatter" in p.lower() or "nano" in p.lower():
        prop = bpy.types.View3DShading.bl_rna.properties.get(p)
        print(f"  shading.{p}: type={prop.type}, desc={prop.description}")

bpy.ops.wm.quit_blender()

