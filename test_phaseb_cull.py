import bpy
import os
import sys
import time
import mathutils
from math import pi

blend_path = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"
out_dir = r"C:\blender-git\scratch\phaseb_cull"
os.makedirs(out_dir, exist_ok=True)

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if args:
    blend_path = args[0]

print(f"[TEST] Loading {blend_path}")
bpy.ops.wm.open_mainfile(filepath=blend_path)

scene = bpy.context.scene

# Dump Tessera-related objects and their properties
print("[TEST] Scene objects with Tessera props:")
for ob in scene.objects:
    keys = [k for k in ob.keys() if 'tessera' in k.lower() or 'scatterflow' in k.lower()]
    if keys:
        print(f"  {ob.name} (type={ob.type}):")
        for k in keys:
            print(f"    {k} = {ob[k]}")

# Render engine: prefer EEVEE Next (material-preview renderer), fall back to legacy
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception:
    scene.render.engine = 'BLENDER_EEVEE'
print(f"[TEST] Engine: {scene.render.engine}")

scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100

cam = scene.camera
if cam is None:
    print("[TEST] No scene camera!")
    sys.exit(1)

def render_frame(camera, tag):
    scene.camera = camera
    out = os.path.join(out_dir, f"phaseb_{tag}.png")
    scene.render.filepath = out
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    dt = time.perf_counter() - t0
    print(f"[TEST] frame '{tag}': {dt:.3f}s -> {out}")

# Camera 1: scene camera as-is
render_frame(cam, "cam0")

# Camera 2: yaw the camera 180 degrees about its own position so most
# scatter instances fall outside the frustum. The C++ side prints
# frame_vis vs inst_mats telemetry when TESSERA_DEBUG=1.
cam2 = bpy.data.objects.new("CullCam", bpy.data.cameras.new("CullCam"))
scene.collection.objects.link(cam2)
cam2.matrix_world = cam.matrix_world.copy()
rot_about_loc = mathutils.Matrix.Translation(cam2.matrix_world.translation) @ mathutils.Matrix.Rotation(pi, 4, 'Z') @ mathutils.Matrix.Translation(-cam2.matrix_world.translation)
cam2.matrix_world = rot_about_loc @ cam2.matrix_world
scene.camera = cam2
render_frame(cam2, "cam_flip")

# Camera 3: pulled in close (near instances visible; far ones should be
# culled by max-draw-distance)
cam3 = bpy.data.objects.new("CullCamNear", bpy.data.cameras.new("CullCamNear"))
scene.collection.objects.link(cam3)
cam3.matrix_world = cam.matrix_world.copy()
cam3.matrix_world.translation = cam3.matrix_world.translation + mathutils.Vector((0, 0, 1.5))
scene.camera = cam3
render_frame(cam3, "cam_near")

print("[TEST] DONE")
