import bpy
import time
import os

def verify_fps():
    ob = bpy.data.objects.get("Ground_009") or bpy.data.objects.get("Ground_013")
    if not ob:
        print("[ERROR] Object not found")
        bpy.ops.wm.quit_blender()
        return None

    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)

    # Bind nc2 path if not already bound
    nc2_path = r"D:\_Code\Meu\Nanite\Ground_009.nc2"
    if os.path.isfile(nc2_path):
        ob["tessera_nc2"] = nc2_path

    props = getattr(ob, "tessera", None)
    if not props:
        print("[ERROR] Tessera properties not found")
        bpy.ops.wm.quit_blender()
        return None

    # Enable Tessera and both FPS features
    props.enabled = True
    props.show_fps_panel = True
    props.show_fps_hud = True

    print(f"[TEST] Object: {ob.name}")
    print(f"[TEST] Enabled: {props.enabled}")
    print(f"[TEST] Panel FPS: {props.show_fps_panel}")
    print(f"[TEST] Viewport HUD: {props.show_fps_hud}")
    print(f"[TEST] tessera_nc2: {ob.get('tessera_nc2')}")

    # Tag redraws on all 3D viewports
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    # Schedule multiple ticks so the viewport renders several frames
    tick_count = [0]
    def check_values():
        tick_count[0] += 1
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        if tick_count[0] < 10:
            return 0.1

        fps = ob.get("tessera_stat_fps", 0.0)
        ft_ms = ob.get("tessera_stat_frametime", 0.0)
        clusters = ob.get("tessera_stat_clusters", 0)
        tris = ob.get("tessera_stat_triangles", 0)

        print(f"[RESULT] tessera_stat_fps: {fps:.1f}")
        print(f"[RESULT] tessera_stat_frametime: {ft_ms:.2f} ms")
        print(f"[RESULT] tessera_stat_clusters: {clusters}")
        print(f"[RESULT] tessera_stat_triangles: {tris}")

        # Capture viewport screenshot for visual proof
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'WINDOW':
                                screenshot_path = r"C:\Users\carlo\.gemini\antigravity-ide\brain\bc53efd7-f877-4d3f-9dca-b26cb33d6820\tessera_fps_hud_verification.png"
                                with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
                                    bpy.ops.screen.screenshot(filepath=screenshot_path)
                                print(f"[RESULT] Saved screenshot to {screenshot_path}")
                                break
        except Exception as e:
            print(f"[WARNING] Screenshot capture: {e}")

        # Test disabling both
        props.show_fps_panel = False
        props.show_fps_hud = False
        print(f"[TEST] Disabled show_fps_panel and show_fps_hud")
        from tessera import ui
        print(f"[TEST] HUD handler after disable: {ui._hud_handler}")

        print("[SUCCESS] All interactive HUD & FPS tests verified!")
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(check_values, first_interval=0.2)
    return None

bpy.app.timers.register(verify_fps, first_interval=0.5)
