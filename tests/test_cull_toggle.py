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

# Now enable Camera Culling on all objects!
print("Enabling Camera Culling (Socket_125 = True) on all modifiers...")
for ob in bpy.data.objects:
    for m in ob.modifiers:
        if m.type == 'NODES' and hasattr(m, 'properties') and hasattr(m.properties, 'inputs'):
            inputs = m.properties.inputs
            s125 = getattr(inputs, 'Socket_125', None)
            if s125:
                s125.value = True

# Re-evaluate depsgraph
depsgraph = bpy.context.evaluated_depsgraph_get()

in_count = 0
out_count = 0
total_count = 0

for inst in depsgraph.object_instances:
    if inst.is_instance:
        total_count += 1
        pos = inst.matrix_world.translation
        if in_frustum(pos):
            in_count += 1
        else:
            out_count += 1

print(f"\nAFTER ENABLING CAMERA CULLING IN MODIFIERS:")
print(f"Total instances: {total_count}")
print(f"Instances INSIDE camera: {in_count}")
print(f"Instances OUTSIDE camera: {out_count}")

