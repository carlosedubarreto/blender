import bpy

for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            rv3d = space.region_3d
            print("=== USER VIEWPORT STATE ===")
            print("view_location:", rv3d.view_location)
            print("view_distance:", rv3d.view_distance)
            print("view_perspective:", rv3d.view_perspective)
            print("view_camera:", space.camera.name if space.camera else None)

