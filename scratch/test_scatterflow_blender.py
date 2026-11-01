import bpy
import sys
import time

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
print("="*80)
print("OPENING SCENE IN SCATTERFLOW BLENDER:")
print("Blend:", blend_path)
print("Blender version:", bpy.app.version_string)
print("Blender binary:", sys.executable)
print("="*80)

bpy.ops.wm.open_mainfile(filepath=blend_path)
scene = bpy.context.scene

print("\n--- ADDONS ENABLED ---")
for addon_name in sorted(bpy.context.preferences.addons.keys()):
    if "scatter" in addon_name.lower() or "nano" in addon_name.lower() or "flow" in addon_name.lower() or "geo" in addon_name.lower():
        print("  Found relevant addon:", addon_name)

print("\n--- SCENE PROPERTIES (search 'scatter', 'nano', 'tessera', 'anno') ---")
for k in scene.keys():
    k_low = k.lower()
    if any(x in k_low for x in ["scatter", "nano", "tessera", "anno", "cull", "lod", "flow"]):
        print(f"  scene['{k}'] = {scene[k]}")

print("\n--- SCENE RNA PROPERTIES ---")
for prop in dir(scene):
    p_low = prop.lower()
    if any(x in p_low for x in ["scatter", "nano", "tessera", "anno", "virtual"]):
        val = getattr(scene, prop, None)
        print(f"  scene.{prop} = {val}")

print("\n--- VIEW3D & SCREEN SETTINGS ---")
for window in bpy.context.window_manager.windows:
    screen = window.screen
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    shading = space.shading
                    print(f"  Shading type: {shading.type}, color_type: {shading.color_type}, light: {shading.light}")
                    for p in dir(space):
                        if any(x in p.lower() for x in ["scatter", "nano", "tessera", "anno", "virtual", "cull"]):
                            print(f"  space.{p} = {getattr(space, p, None)}")
                    for p in dir(shading):
                        if any(x in p.lower() for x in ["scatter", "nano", "tessera", "anno", "virtual"]):
                            print(f"  shading.{p} = {getattr(shading, p, None)}")

print("\n--- OBJECT PROPERTIES ---")
scatter_objs = []
for ob in scene.objects:
    for k in ob.keys():
        if any(x in k.lower() for x in ["scatter", "nano", "tessera", "anno"]):
            print(f"  ob '{ob.name}'['{k}'] = {ob[k]}")
            scatter_objs.append(ob)
            break

print("\n--- MODIFIERS ON SCATTER OBJECTS ---")
for ob in scene.objects:
    if ob.modifiers:
        mods = [f"{m.name} ({m.type})" for m in ob.modifiers]
        if any("gn" in m.lower() or "scatter" in m.lower() or "node" in m.lower() for m in mods):
            print(f"  Object '{ob.name}': {mods}")
            for m in ob.modifiers:
                if m.type == 'NODES' and m.node_group:
                    print(f"    NodeGroup: {m.node_group.name}")
                    for inp in m.node_group.inputs if hasattr(m.node_group, 'inputs') else []:
                        pass

print("="*80)
bpy.ops.wm.quit_blender()

