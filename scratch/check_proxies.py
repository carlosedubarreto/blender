import bpy

for ob in bpy.context.scene.objects:
    if 'OSGN' in ob.name or 'Proxy' in ob.name:
        print(f"{ob.name}: loc={ob.location}, scale={ob.scale}, hide_vp={ob.hide_viewport}")

