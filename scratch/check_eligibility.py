import bpy
import tessera.ui

scene = bpy.context.scene
print("=== CHECKING OBJECT ELIGIBILITY & TESSERA STATUS ===")
enabled_count = 0
fallback_count = 0

for ob in scene.objects:
    if ob.type != 'MESH':
        continue
    is_en = ob.get("tessera_enabled", False)
    nc2 = tessera.ui.get_existing_nc2_path(ob)
    if is_en and nc2:
        enabled_count += 1
    else:
        fallback_count += 1
        print(f"FALLBACK NATIVE MESH: {ob.name} (enabled={is_en}, nc2={nc2})")

print(f"Total: {enabled_count} Tessera Enabled, {fallback_count} Native Fallback")

