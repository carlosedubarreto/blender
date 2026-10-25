import bpy
import mathutils
import math
import os

blend_path = r"D:\_Code\Meu\Nanite\Tessera_Gravel_13_color_Test.blend"
out_dir = r"C:\Users\carlo\.gemini\antigravity-ide\brain\bc53efd7-f877-4d3f-9dca-b26cb33d6820\scratch"
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=blend_path)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

gravel = bpy.data.objects.get("Gravel_13")
print("Gravel_13 object found:", gravel is not None)
if gravel:
    print(f"Location: {gravel.location}, Scale: {gravel.scale}")
    # Check Tessera properties
    for k in gravel.keys():
        if 'tessera' in k.lower():
            print(f"  {k} = {gravel[k]}")

# Position camera for close-up inspection of gravel surface
cam = scene.camera
if not cam:
    cam_data = bpy.data.cameras.new("InspectCam")
    cam = bpy.data.objects.new("InspectCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

cam.location = mathutils.Vector((0.0, -1.8, 0.9))
cam.rotation_euler = (math.radians(65), 0, 0)

# 1. Render under Sun 1
sun = bpy.data.objects.get("Sun")
if not sun:
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            sun = obj
            break

if sun:
    sun.rotation_euler = (math.radians(45), math.radians(30), math.radians(20))

scene.render.filepath = os.path.join(out_dir, "verify_tessera_sun1.png")
bpy.ops.render.render(write_still=True)
print(f"Saved: {scene.render.filepath}")

# 2. Rotate sun to grazing angle to stress-test normal mapping & cluster boundaries
if sun:
    sun.rotation_euler = (math.radians(75), math.radians(-45), math.radians(110))

scene.render.filepath = os.path.join(out_dir, "verify_tessera_sun2.png")
bpy.ops.render.render(write_still=True)
print(f"Saved: {scene.render.filepath}")

# Check telemetry
if hasattr(bpy.types, "TESSERA_PT_telemetry") or hasattr(bpy.context.window_manager, "tessera_telemetry"):
    print("Telemetry module registered")

print("Verification render completed successfully.")
