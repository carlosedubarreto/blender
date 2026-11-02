import bpy

cam = bpy.context.scene.camera
print(f"Camera loc: {cam.location}, rot: {cam.rotation_euler}")

