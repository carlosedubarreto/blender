import bpy
import numpy as np

depsgraph = bpy.context.evaluated_depsgraph_get()
cam = bpy.data.objects.get("Camera")

render = bpy.context.scene.render
winx = (render.resolution_x * render.resolution_percentage) // 100
winy = (render.resolution_y * render.resolution_percentage) // 100

viewmat = cam.matrix_world.inverted()
winmat = cam.calc_matrix_camera(depsgraph, x=winx, y=winy)
viewProj = np.array(winmat @ viewmat)

r0, r1, r2, r3 = viewProj[0], viewProj[1], viewProj[2], viewProj[3]
planes = np.zeros((6, 4))
planes[0] = r3 + r0
planes[1] = r3 - r0
planes[2] = r3 + r1
planes[3] = r3 - r1
planes[4] = r3 + r2
planes[5] = r3 - r2
for p in range(6):
    planes[p] /= np.linalg.norm(planes[p, :3])

def in_frustum(pos):
    c = np.array([pos[0], pos[1], pos[2], 1.0])
    for p in range(6):
        if np.dot(planes[p], c) < -0.1:
            return False
    return True

# Check per emitter object
by_emitter = {}
for inst in depsgraph.object_instances:
    if inst.is_instance:
        parent = inst.parent.name if inst.parent else "None"
        if parent not in by_emitter:
            by_emitter[parent] = {"in": 0, "out": 0}
        pos = inst.matrix_world.translation
        if in_frustum(pos):
            by_emitter[parent]["in"] += 1
        else:
            by_emitter[parent]["out"] += 1

print("\n=== INSTANCES BY EMITTER (IN vs OUT of Camera Frustum) ===")
for name, counts in sorted(by_emitter.items()):
    print(f"Emitter '{name}': IN={counts['in']}, OUT={counts['out']}, TOTAL={counts['in'] + counts['out']}")

