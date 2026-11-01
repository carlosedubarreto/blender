import bpy

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

ob = bpy.data.objects['Rocks_Small']
print("tessera_min_screen_radius in ob?", 'tessera_min_screen_radius' in ob)
print("tessera_batch_min_screen_radius in scene?", 'tessera_batch_min_screen_radius' in bpy.context.scene)

# Let's calculate screen_radius_px for Rocks_Small:
# Radius of Desert Rock2_V1 source mesh:
import math
# Source mesh radius is ~0.8m
# Rock instance scale: median is 0.05
# world_radius = 0.8 * 0.05 = 0.04m
# dist = 6.0m
# px_scale = 1300
# screen_radius_px = (0.04 * 1300) / 6.0 = 8.6 pixels!
# 8.6 pixels > 3.0 pixels!
print("Calculated screen_radius_px is ~8.6px, which is > 3.0px!")

bpy.ops.wm.quit_blender()

