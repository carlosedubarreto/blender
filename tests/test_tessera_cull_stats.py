import bpy

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'

# Enable Tessera
for ob in bpy.data.objects:
    if "SweetViolet" in ob.name or "Debris" in ob.name or "Grass" in ob.name:
        ob["tessera_enabled"] = 1
        if hasattr(ob, "tessera"):
            ob.tessera.enabled = True

# Render with Camera
scene.camera = bpy.data.objects.get("Camera")
bpy.ops.render.render()

print("\n=== CAMERA VIEW RENDER STATS ===")
for ob in bpy.data.objects:
    if ob.get("tessera_nc2") or ob.get("tessera_nc2_path"):
        drawn = ob.get("tessera_stat_drawn_instances", -1)
        total = ob.get("tessera_stat_total_instances", -1)
        clusters = ob.get("tessera_stat_clusters", -1)
        print(f"Object '{ob.name}': drawn={drawn}, total={total}, clusters={clusters}")

