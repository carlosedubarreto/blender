import bpy
import mathutils
import numpy as np

depsgraph = bpy.context.evaluated_depsgraph_get()
cam = bpy.data.objects.get("Camera")
scene = bpy.context.scene
render = scene.render
winx = (render.resolution_x * render.resolution_percentage) // 100
winy = (render.resolution_y * render.resolution_percentage) // 100

viewmat = cam.matrix_world.inverted()
winmat = cam.calc_matrix_camera(depsgraph, x=winx, y=winy)
viewProj = winmat @ viewmat

# Extract frustum planes exactly like extract_frustum_planes in tessera_lod.cc:
# In C++:
# const auto row = [&](int r) { return float4(m[0][r], m[1][r], m[2][r], m[3][r]); };
# planes[0] = r3 + r0; // left
# planes[1] = r3 - r0; // right
# planes[2] = r3 + r1; // bottom
# planes[3] = r3 - r1; // top
# planes[4] = r3 + r2; // near
# planes[5] = r3 - r2; // far

m = np.array(viewProj) # 4x4 matrix
r0 = m[0, :]
r1 = m[1, :]
r2 = m[2, :]
r3 = m[3, :]

planes = np.zeros((6, 4))
planes[0] = r3 + r0 # left
planes[1] = r3 - r0 # right
planes[2] = r3 + r1 # bottom
planes[3] = r3 - r1 # top
planes[4] = r3 + r2 # near
planes[5] = r3 - r2 # far

# Normalize plane normals
for p in range(6):
    n = planes[p, :3]
    length = np.linalg.norm(n)
    if length > 0:
        planes[p] /= length

def sphere_in_frustum(center, radius, margin=0.0):
    c = np.array([center[0], center[1], center[2], 1.0])
    for p in range(6):
        dist = np.dot(planes[p], c)
        if dist < -(radius + margin):
            return False, p, dist
    return True, -1, 0.0

# Now test instances
passed = 0
failed = 0
fail_planes = [0]*6

tested = 0
for inst in depsgraph.object_instances:
    if inst.is_instance:
        tested += 1
        pos = inst.matrix_world.translation
        in_f, p_idx, dist = sphere_in_frustum(pos, 0.1)
        if in_f:
            passed += 1
        else:
            failed += 1
            fail_planes[p_idx] += 1

print(f"Total instances tested: {tested}")
print(f"Passed frustum: {passed}")
print(f"Failed frustum: {failed}")
print(f"Failed by planes [left, right, bottom, top, near, far]: {fail_planes}")

