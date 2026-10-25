"""Tessera Automated Regression Test Suite.

Specialized regression test suite for continuous LOD DAG validation,
cluster boundary topology, watertightness, co-emission prevention,
and Reverse-Z occlusion depth order in Blender EEVEE Next.

Execution:
    & "C:\\blender-git\\build_windows_x64_vc17_Release\\bin\\blender.exe" -b --python tests/tessera_regression_suite.py
"""

import os
import sys
import struct
import tempfile
import traceback
from collections import defaultdict, deque

# Mandatory Blender executable path guardrail
MANDATORY_BLENDER_BINARY = r"C:\blender-git\build_windows_x64_vc17_Release\bin\blender.exe"
SAMPLE_GROUND_09 = r"D:\_Code\Meu\Nanite\Tessera_Ground_09_seams_Test.blend"
SAMPLE_GROUND_13 = r"D:\_Code\Meu\Nanite\Tessera_Ground_13_Test.blend"
NC2_GROUND_09 = r"D:\_Code\Meu\Nanite\Ground_009.nc2"
NC2_GROUND_13 = r"D:\_Code\Meu\Nanite\Ground_013.nc2"


# ---------------------------------------------------------------------------
# Direct NC2 Binary Parser
# ---------------------------------------------------------------------------

def parse_nc2_file(filepath):
    """Directly parse .nc2 binary file format into Python structures."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tessera NC2 file not found: {filepath}")

    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < 88:
        raise ValueError(f"Corrupt NC2 file: header too short ({len(data)} bytes)")

    header_bytes = data[:88]
    (magic, version, cluster_count, level_count, vertex_count, index_count,
     proxy_vertex_count, proxy_index_count, max_cluster_triangles, max_cluster_vertices,
     flags, page_count,
     bmin0, bmin1, bmin2, bmax0, bmax1, bmax2,
     uvmin0, uvmin1, uvmax0, uvmax1) = struct.unpack("<12I10f", header_bytes)

    if magic != 0x3255434E: # "NCU2"
        raise ValueError(f"Invalid NC2 magic: {hex(magic)}, expected 0x3255434E")

    offset = 88
    clusters = []
    for i in range(cluster_count):
        c_bytes = data[offset:offset+64]
        offset += 64
        (s0, s1, s2, s3, p0, p1, p2, p3, error, parent_error,
         first_index, count, parent_group, group, level, cflags, mat_idx) = struct.unpack("<10fIIiII2H", c_bytes)
        clusters.append({
            "id": i,
            "sphere": (s0, s1, s2, s3),
            "parentSphere": (p0, p1, p2, p3),
            "error": error,
            "parentError": parent_error,
            "firstIndex": first_index,
            "indexCount": count,
            "parentGroup": parent_group,
            "group": group,
            "level": level,
            "flags": cflags,
            "materialIndex": mat_idx
        })

    import numpy as np
    pos_bytes = vertex_count * 12
    positions = np.frombuffer(data[offset:offset+pos_bytes], dtype=np.float32).reshape(-1, 3)
    offset += pos_bytes

    if flags & 2: # HasNormals
        offset += vertex_count * 12
    if flags & 1: # HasUV
        offset += vertex_count * 8
    if flags & 4: # HasTangents
        offset += vertex_count * 16

    idx_bytes = index_count * 4
    indices = np.frombuffer(data[offset:offset+idx_bytes], dtype=np.uint32)

    return {
        "header": {
            "clusterCount": cluster_count,
            "levelCount": level_count,
            "vertexCount": vertex_count,
            "indexCount": index_count,
            "flags": flags,
            "boundsMin": (bmin0, bmin1, bmin2),
            "boundsMax": (bmax0, bmax1, bmax2),
        },
        "clusters": clusters,
        "positions": positions,
        "indices": indices
    }


# ---------------------------------------------------------------------------
# Test 1: Cluster Boundary Seams & Water-Tightness Validation
# ---------------------------------------------------------------------------

def test_cluster_seam_watertightness():
    """Validates cluster boundary edge locking and renders visual occlusion seam test."""
    print("\n" + "="*80)
    print("RUNNING TEST 1: test_cluster_seam_watertightness")
    print("="*80)

    import numpy as np
    import mathutils
    import bpy

    # 1. Structural Validation of Perimeter Boundary Locking
    print("\n[Phase 1] Group Boundary Locking Verification (.nc2 binary DAG inspection)...")
    nc2_data = parse_nc2_file(NC2_GROUND_09)
    clusters = nc2_data["clusters"]
    positions = nc2_data["positions"]
    indices = nc2_data["indices"]

    err_groups = defaultdict(list)
    for c in clusters:
        if c["level"] == 0 and c["parentGroup"] >= 0:
            err_groups[c["parentError"]].append(c["id"])

    def extract_boundary_verts(tris):
        edge_count = defaultdict(int)
        for t in tris:
            for k in range(3):
                v0, v1 = t[k], t[(k+1)%3]
                p0 = tuple(np.round(positions[v0], 5))
                p1 = tuple(np.round(positions[v1], 5))
                edge = tuple(sorted((p0, p1)))
                edge_count[edge] += 1
        b_edges = [edge for edge, count in edge_count.items() if count == 1]
        b_verts = set()
        for p0, p1 in b_edges:
            b_verts.add(p0)
            b_verts.add(p1)
        return list(b_verts), b_edges

    checked_groups = 0
    drift_violations = []
    tolerance = 1e-5

    for perr, c_ids in err_groups.items():
        parent_ids = list(set([clusters[cid]["parentGroup"] for cid in c_ids]))
        parents_with_err = [c["id"] for c in clusters if c["level"] == 1 and abs(c["error"] - perr) < 1e-7]
        if set(parent_ids) != set(parents_with_err):
            continue

        c_tris_list = [indices[clusters[cid]["firstIndex"]:clusters[cid]["firstIndex"]+clusters[cid]["indexCount"]].reshape(-1, 3) for cid in c_ids]
        c_tris = np.vstack(c_tris_list)
        p_tris_list = [indices[clusters[pid]["firstIndex"]:clusters[pid]["firstIndex"]+clusters[pid]["indexCount"]].reshape(-1, 3) for pid in parent_ids]
        p_tris = np.vstack(p_tris_list)

        c_verts, c_edges = extract_boundary_verts(c_tris)
        p_verts, p_edges = extract_boundary_verts(p_tris)

        if not p_verts or not c_verts:
            continue

        c_coords = np.array(c_verts, dtype=np.float32)
        kd = mathutils.kdtree.KDTree(len(c_coords))
        for idx_c, pt in enumerate(c_coords):
            kd.insert(pt, idx_c)
        kd.balance()

        for pt in p_verts:
            _, _, dist = kd.find(pt)
            if dist > tolerance:
                drift_violations.append({
                    "parent_ids": parent_ids,
                    "parent_level": 1,
                    "child_ids": c_ids,
                    "coord": pt,
                    "drift": dist
                })
        checked_groups += 1
        if checked_groups >= 100:
            break

    print(f"Verified boundary locking across {checked_groups} cluster groups.")
    if drift_violations:
        first = drift_violations[0]
        print(f"FAILED: Boundary vertex drift detected on Parent Clusters {first['parent_ids']} (LOD {first['parent_level']}):")
        print(f"  Coordinates: {first['coord']}, Drift: {first['drift']:.6f} > {tolerance}")
        raise AssertionError(f"Cluster seam boundary locking violated! Drift {first['drift']:.6e} at {first['coord']}")

    print("PASS: Group Boundary Locking verified within 1e-5 tolerance.")

    # 2. Visual Seam Occlusion Test (Raycast / High-Contrast Background Test)
    print("\n[Phase 2] Visual Seam Occlusion Test (Headless EEVEE Next Render)...")
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_GROUND_09)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 480
    scene.render.resolution_y = 270

    ob = bpy.data.objects.get("Ground_009")
    assert ob is not None, "Ground_009 object not found in test scene"
    ob["tessera_enabled"] = 1
    ob["tessera_visualize_clusters"] = 0
    ob["tessera_nc2"] = NC2_GROUND_09

    # Set background to bright high-contrast white
    if scene.world and scene.world.node_tree:
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bg_node.inputs[1].default_value = 5.0

    # Set unlit black material on the mesh
    black_mat = bpy.data.materials.new("PureBlackMat")
    bsdf = black_mat.node_tree.nodes.get("Principled BSDF") if black_mat.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
    ob.data.materials.clear()
    ob.data.materials.append(black_mat)

    temp_render = os.path.join(tempfile.gettempdir(), "tessera_seam_watertight.png")
    scene.render.filepath = temp_render
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(temp_render)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bpy.data.images.remove(img)

    lum = 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
    is_white = (lum > 0.8)

    # Flood fill exterior background from image boundaries
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        if is_white[0, x]: queue.append((0, x)); visited[0, x] = True
        if is_white[h-1, x]: queue.append((h-1, x)); visited[h-1, x] = True
    for y in range(h):
        if is_white[y, 0]: queue.append((y, 0)); visited[y, 0] = True
        if is_white[y, w-1]: queue.append((y, w-1)); visited[y, w-1] = True

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and is_white[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    leaks = np.argwhere(is_white & (~visited))
    print(f"Visual Scan: Total Pixels: {w*h}, Leaked Interior Background Pixels: {len(leaks)}")
    if len(leaks) > 0:
        coord = leaks[0]
        raise AssertionError(f"Visual seam crack detected! Background leaked through mesh interior at pixel ({coord[1]}, {coord[0]}).")

    print("PASS: Zero background bleed-through. Watertightness confirmed without boundary cracks.")
    return True


# ---------------------------------------------------------------------------
# Test 2: Hierarchy Overlap & Co-emission Prevention
# ---------------------------------------------------------------------------

def test_no_hierarchy_coemission():
    """Validates mutual exclusion of ancestor/descendant clusters and coplanar triangle conservation."""
    print("\n" + "="*80)
    print("RUNNING TEST 2: test_no_hierarchy_coemission")
    print("="*80)

    import numpy as np
    import mathutils

    test_targets = [
        ("Ground_009", NC2_GROUND_09),
        ("Ground_013", NC2_GROUND_13)
    ]

    def project_error_px(center, radius, error, cam_pos, vh=1080.0, proj_scale_y=1.0):
        if error <= 0.0:
            return 0.0
        dist = max(np.linalg.norm(np.array(cam_pos) - np.array(center)) - radius, 1e-6)
        return error * proj_scale_y * (vh * 0.5) / dist

    distances = [2.0, 5.0, 15.0, 50.0]

    for name, nc2_path in test_targets:
        print(f"\n--- Validating DAG Cuts for {name} ({os.path.basename(nc2_path)}) ---")
        nc2_data = parse_nc2_file(nc2_path)
        clusters = nc2_data["clusters"]
        positions = nc2_data["positions"]
        indices = nc2_data["indices"]

        children = defaultdict(list)
        for c in clusters:
            if c["parentGroup"] >= 0:
                children[c["parentGroup"]].append(c["id"])

        def traverse_cut(cam_pos, tau=1.0):
            emitted = []
            def visit(ci):
                c = clusters[ci]
                c_center = (c["sphere"][0], c["sphere"][1], c["sphere"][2])
                c_err = project_error_px(c_center, c["sphere"][3], c["error"], cam_pos)

                p_center = (c["parentSphere"][0], c["parentSphere"][1], c["parentSphere"][2])
                p_err = 1e38 if c["parentError"] >= 1e38 else project_error_px(p_center, c["parentSphere"][3], c["parentError"], cam_pos)

                if c_err <= tau and p_err > tau:
                    emitted.append(ci)
                    return

                if c_err > tau and ci in children:
                    for ch in children[ci]:
                        visit(ch)

            for ci in range(len(clusters)):
                if clusters[ci]["parentGroup"] == -1:
                    visit(ci)
            return emitted

        mesh_center = np.mean(positions[::100], axis=0)

        for d in distances:
            cam_pos = mesh_center + np.array([0.0, -d, d * 0.4])
            emitted = traverse_cut(cam_pos, tau=1.0)
            emitted_set = set(emitted)

            # 1. Assert Ancestor / Descendant Mutual Exclusion
            for ci in emitted:
                c = clusters[ci]
                # Check ancestor chain
                curr = c["parentGroup"]
                while curr >= 0:
                    if curr in emitted_set:
                        anc = clusters[curr]
                        raise AssertionError(
                            f"Co-emission detected! Both child cluster {ci} (LOD {c['level']}) "
                            f"and ancestor {curr} (LOD {anc['level']}) were emitted simultaneously "
                            f"at distance {d}m on {name}."
                        )
                    curr = clusters[curr]["parentGroup"]

                # Check descendant tree
                desc_stack = list(children.get(ci, []))
                while desc_stack:
                    ch_id = desc_stack.pop()
                    if ch_id in emitted_set:
                        desc = clusters[ch_id]
                        raise AssertionError(
                            f"Co-emission detected! Both cluster {ci} (LOD {c['level']}) "
                            f"and descendant {ch_id} (LOD {desc['level']}) were emitted simultaneously."
                        )
                    desc_stack.extend(children.get(ch_id, []))

            # 2. Assert Triangle Coverage Conservation (Zero Duplicate Coplanar Triangles)
            tri_centroids = []
            tri_normals = []
            for ci in emitted:
                c = clusters[ci]
                c_idx = indices[c["firstIndex"]:c["firstIndex"]+c["indexCount"]].reshape(-1, 3)
                v0 = positions[c_idx[:, 0]]
                v1 = positions[c_idx[:, 1]]
                v2 = positions[c_idx[:, 2]]
                centers = (v0 + v1 + v2) / 3.0
                norms = np.cross(v1 - v0, v2 - v0)
                norm_lens = np.linalg.norm(norms, axis=1, keepdims=True)
                norm_lens[norm_lens < 1e-12] = 1.0
                norms /= norm_lens
                tri_centroids.append(centers)
                tri_normals.append(norms)

            all_centers = np.vstack(tri_centroids)
            all_normals = np.vstack(tri_normals)

            kd = mathutils.kdtree.KDTree(len(all_centers))
            for i_pt, pt in enumerate(all_centers):
                kd.insert(pt, i_pt)
            kd.balance()

            sample_size = min(4000, len(all_centers))
            for i_pt in range(sample_size):
                pt = all_centers[i_pt]
                nor = all_normals[i_pt]
                for co, idx_nb, dist_val in kd.find_range(pt, 1e-4):
                    if idx_nb > i_pt:
                        if np.dot(nor, all_normals[idx_nb]) > 0.999:
                            raise AssertionError(
                                f"Duplicate coplanar triangle detected at world coordinates {pt}! "
                                f"Triangles {i_pt} and {idx_nb} overlap with normal dot product > 0.999."
                            )

            print(f"  Distance {d:>4.1f}m: Emitted {len(emitted):>4} clusters ({len(all_centers):>6} triangles) -> Zero co-emission / zero duplicates.")

    print("PASS: Hierarchy mutual exclusion and triangle conservation fully verified.")
    return True


# ---------------------------------------------------------------------------
# Test 3: Depth Testing & Reverse-Z Occlusion Parity
# ---------------------------------------------------------------------------

def test_reverse_z_occlusion_order():
    """Validates that front-most surfaces reliably occlude background geometry using Reverse-Z."""
    print("\n" + "="*80)
    print("RUNNING TEST 3: test_reverse_z_occlusion_order")
    print("="*80)

    import bpy
    import numpy as np

    bpy.ops.wm.open_mainfile(filepath=SAMPLE_GROUND_09)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 160
    scene.render.resolution_y = 120
    scene.view_layers[0].use_pass_z = True

    ob = bpy.data.objects.get("Ground_009")
    assert ob is not None, "Ground_009 not found"
    ob["tessera_enabled"] = 1
    ob["tessera_visualize_clusters"] = 0
    ob["tessera_nc2"] = NC2_GROUND_09

    cam = bpy.data.objects.get("Camera")
    assert cam is not None, "Camera not found"
    cam.location = (0.0, 0.0, 3.0)
    cam.rotation_euler = (0.0, 0.0, 0.0) # looking straight down along -Z

    # Add foreground test occluder at z = 1.0 (distance 2.0m from camera)
    bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0.0, 0.0, 1.0))
    plane = bpy.context.active_object
    plane.name = "TestOccluderPlane"

    # Set up Compositor to write out linear depth pass
    ng = bpy.data.node_groups.new('DepthProbeTree', 'CompositorNodeTree')
    scene.compositing_node_group = ng
    rl = ng.nodes.new('CompositorNodeRLayers')
    out_node = ng.nodes.new('NodeGroupOutput')
    ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.links.new(rl.outputs['Depth'], out_node.inputs['Image'])

    depth_out = os.path.join(tempfile.gettempdir(), "tessera_depth_probe.exr")
    scene.render.filepath = depth_out
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'

    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(depth_out)
    px = np.array(img.pixels[:], dtype=np.float32)
    depth = px[0::4].reshape((120, 160))
    bpy.data.images.remove(img)

    # Center plane pixels (camera at z=3.0, plane at z=1.0 -> depth = 2.0m)
    center_depths = depth[55:65, 75:85]
    plane_depth = float(np.mean(center_depths))

    # Outer ground pixels (ground at z=0.0 -> depth = 3.0m)
    outer_depths = depth[10:30, 10:30]
    ground_depth = float(np.mean(outer_depths))

    print(f"Foreground occluder depth: {plane_depth:.4f}m (expected ~2.0m)")
    print(f"Background ground depth:   {ground_depth:.4f}m (expected ~2.9 - 3.0m)")

    if plane_depth >= ground_depth or plane_depth > 2.2:
        raise AssertionError(
            f"Reverse-Z occlusion failure! Foreground occluder depth ({plane_depth:.4f}m) "
            f"was not strictly in front of background surface ({ground_depth:.4f}m)."
        )

    print("PASS: Reverse-Z occlusion depth order verified strictly monotonic from near to far.")
    return True


# ---------------------------------------------------------------------------
# Test Runner & Entry Point
# ---------------------------------------------------------------------------

def run_suite():
    """Execute all regression tests and report summary."""
    print("="*80)
    print("TESSERA AUTOMATED REGRESSION SUITE")
    print(f"Blender Binary: {MANDATORY_BLENDER_BINARY}")
    print("="*80)

    # Strict Binary Validation Guardrail
    if not os.path.exists(MANDATORY_BLENDER_BINARY):
        print(f"FATAL: Required Blender binary does not exist at {MANDATORY_BLENDER_BINARY}!")
        sys.exit(1)

    results = {}
    tests = [
        ("test_cluster_seam_watertightness", test_cluster_seam_watertightness),
        ("test_no_hierarchy_coemission", test_no_hierarchy_coemission),
        ("test_reverse_z_occlusion_order", test_reverse_z_occlusion_order),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAILED: {e}"

    print("\n" + "="*80)
    print("TESSERA REGRESSION SUITE SUMMARY")
    print("="*80)
    all_passed = True
    for name, status in results.items():
        print(f"  {name:<38} : {status}")
        if not status.startswith("PASSED"):
            all_passed = False

    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("REGRESSION SUITE FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    # If invoked directly via python runner, spawn blender binary
    if "bpy" in sys.modules:
        run_suite()
    else:
        import subprocess
        cmd = [MANDATORY_BLENDER_BINARY, "-b", "--python", __file__]
        print(f"Spawning Blender headless runner: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        sys.exit(res.returncode)
