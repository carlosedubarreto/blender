import os
import sys
import time
import bpy

blend_path = r"D:\_Code\Meu\Nanite\Tessera_Ground_13_Test.blend"
if not os.path.isfile(blend_path):
    blend_path = r"D:\_Code\Meu\Nanite\Tessera_Ground_09_Test.blend"

print(f"Loading blend: {blend_path}")
bpy.ops.wm.open_mainfile(filepath=blend_path)

ob = bpy.data.objects.get("Ground_013") or bpy.data.objects.get("Ground_009")
print(f"Testing object: {ob.name if ob else 'None'}")
assert ob is not None, "Target object not found!"

props = getattr(ob, "tessera", None)
assert props is not None, "Tessera properties not found on object!"

# Ensure tessera enabled
props.enabled = True
print(f"Tessera enabled: {ob.get('tessera_enabled')}")

from tessera import ui, render

print("Initial properties:")
print(f"  show_fps_panel: {props.show_fps_panel}")
print(f"  show_fps_hud: {props.show_fps_hud}")

# Test 1: Zero-cost verification when disabled
print("\n--- TEST 1: Zero-cost inactive mode ---")
props.show_fps_panel = False
props.show_fps_hud = False
assert ui._hud_handler is None, "HUD handler should be None when show_fps_hud is False"
print("PASS: Inactive mode bypasses HUD draw handler.")

# Test 2: Enable show_fps_panel
print("\n--- TEST 2: Panel FPS mode ---")
props.show_fps_panel = True
assert ob.get("tessera_show_fps_panel") == 1, "tessera_show_fps_panel IDProperty not synced!"

# Force a redraw pass to simulate viewport frames
for i in range(5):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    bpy.context.view_layer.update()
    time.sleep(0.02)

print(f"  tessera_stat_fps: {ob.get('tessera_stat_fps')}")
print(f"  tessera_stat_frametime: {ob.get('tessera_stat_frametime')}")

# Test 3: Enable show_fps_hud
print("\n--- TEST 3: Viewport HUD overlay mode ---")
props.show_fps_hud = True
assert ob.get("tessera_show_fps_hud") == 1, "tessera_show_fps_hud IDProperty not synced!"
assert ui._hud_handler is not None, "HUD handler should be registered when show_fps_hud is True!"

# Simulate calling draw_tessera_hud directly to test for any runtime errors
bpy.context.view_layer.objects.active = ob
try:
    ui.draw_tessera_hud()
    print("PASS: draw_tessera_hud executed with zero errors.")
except Exception as e:
    print(f"FAIL: draw_tessera_hud raised {e}")
    raise

# Test 4: Disable both and verify unregistration
print("\n--- TEST 4: Disable both toggles ---")
props.show_fps_panel = False
props.show_fps_hud = False
assert ui._hud_handler is None, "HUD handler must be unregistered when show_fps_hud is disabled!"
print("PASS: Zero-cost inactive mode successfully unregisters handler.")

print("\nALL FPS & HUD TELEMETRY TESTS PASSED SUCCESSFULLY!")
