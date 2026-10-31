import time
import bpy

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

t0 = time.perf_counter()
for i in range(10):
    scene.frame_set(scene.frame_current)
    depsgraph.update()
t1 = time.perf_counter()
print(f"Depsgraph update time: {(t1 - t0) * 100.0:.2f} ms per update")

