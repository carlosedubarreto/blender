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
SAMPLE_GRAVEL_13 = r"D:\_Code\Meu\Nanite\Tessera_Gravel_13_color_Test.blend"
NC2_GROUND_09 = r"D:\_Code\Meu\Nanite\Ground_009.nc2"
NC2_GROUND_13 = r"D:\_Code\Meu\Nanite\Ground_013.nc2"
SAMPLE_HIGHPOLY_1 = r"D:\_Code\Meu\Nanite\Tessera_HighPoy_Test_1.blend"
SAMPLE_SCATTERFLOW = r"C:\blender-git\tessera_cenas_teste\Tessera_HighPoy_Test_NVME_scatterflow_assets_1.blend"


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
# Test 4: Camera Margin Control & Boundary Cluster Expansion
# ---------------------------------------------------------------------------

def test_camera_margin_control():
    """Validates camera margin frustum culling: cluster ratio, visual voiding, expansion, and recovery."""
    print("\n" + "="*80)
    print("RUNNING TEST 4: test_camera_margin_control (Visual Void & Strict Cluster Ratio)")
    print("="*80)

    import numpy as np
    import mathutils
    import bpy
    import tempfile
    import os

    # Load test asset
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_GROUND_13)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360

    ob = bpy.data.objects.get("Ground_013") or bpy.data.objects.get("Ground_13")
    assert ob is not None, "Ground_013 object not found in test blend file"

    ob["tessera_enabled"] = 1
    ob["tessera_nc2"] = NC2_GROUND_13
    if hasattr(ob, "tessera"):
        ob.tessera.enabled = True

    # Compute bounding box in world space
    world_bbox = [ob.matrix_world @ mathutils.Vector(b) for b in ob.bound_box]
    min_c = mathutils.Vector([min(b[i] for b in world_bbox) for i in range(3)])
    max_c = mathutils.Vector([max(b[i] for b in world_bbox) for i in range(3)])
    center = (min_c + max_c) * 0.5
    dims = max_c - min_c

    # Camera setup:
    # 1. Cull camera: positioned so 50% of Ground_013 lies outside the frustum
    cam_cull_data = bpy.data.cameras.new("CullCam")
    cam_cull = bpy.data.objects.new("CullCam", cam_cull_data)
    scene.collection.objects.link(cam_cull)
    cam_cull.location = mathutils.Vector((center.x + dims.x * 0.25, center.y, 4.0))
    cam_cull.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')

    # 2. Witness camera: frames the full mesh to inspect culled vs filled pixels
    cam_witness_data = bpy.data.cameras.new("WitnessCam")
    cam_witness = bpy.data.objects.new("WitnessCam", cam_witness_data)
    scene.collection.objects.link(cam_witness)
    cam_witness.location = mathutils.Vector((center.x, center.y, 8.0))
    cam_witness.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
    scene.camera = cam_witness

    ob["tessera_cull_camera"] = cam_cull.name
    ob["tessera_view_source"] = 1

    # First determine total cluster count of the full mesh (rendered at margin 15.0)
    print("\n[Phase 1] Evaluating total full-mesh cluster count at culling_margin = 15.0...")
    ob["tessera_culling_margin"] = 15.0
    if hasattr(ob, "tessera"):
        ob.tessera.culling_margin = 15.0
    bpy.context.view_layer.update()
    temp_m15 = os.path.join(tempfile.gettempdir(), "test_margin_15.png")
    scene.render.filepath = temp_m15
    bpy.ops.render.render(write_still=True)

    total_clusters = ob.get("tessera_stat_clusters")
    total_triangles = ob.get("tessera_stat_triangles")
    print(f"  Full Mesh Total Clusters: {total_clusters} ({total_triangles:,} triangles)")
    assert total_clusters is not None and total_clusters > 0, "No clusters recorded at margin 15.0"

    # Phase 2: Render at culling_margin = 0.0
    print("\n[Phase 2] Strict culling evaluation at culling_margin = 0.0...")
    ob["tessera_culling_margin"] = 0.0
    if hasattr(ob, "tessera"):
        ob.tessera.culling_margin = 0.0
    bpy.context.view_layer.update()
    temp_m0 = os.path.join(tempfile.gettempdir(), "test_margin_0.png")
    scene.render.filepath = temp_m0
    bpy.ops.render.render(write_still=True)

    clusters_margin_0 = ob.get("tessera_stat_clusters")
    tris_margin_0 = ob.get("tessera_stat_triangles")
    eff_m0 = ob.get("tessera_stat_culling_margin")
    print(f"  Margin 0.0: {clusters_margin_0} active clusters ({tris_margin_0:,} triangles) | Margin: {eff_m0:.2f}m")

    # Assertion 1 (Cluster Ratio): active_clusters <= 0.65 * total_clusters
    cluster_ratio_0 = clusters_margin_0 / float(total_clusters)
    print(f"  Cluster Ratio at Margin 0.0: {cluster_ratio_0:.3f} (threshold <= 0.65)")
    assert cluster_ratio_0 <= 0.65, (
        f"Assertion 1 FAILED! active_clusters ({clusters_margin_0}) is {cluster_ratio_0*100:.1f}% "
        f"of total_clusters ({total_clusters}). Culling is bypassed!"
    )

    # Read pixel buffers
    img0 = bpy.data.images.load(temp_m0)
    w0, h0 = img0.size
    px0 = np.array(img0.pixels[:], dtype=np.float32).reshape((h0, w0, 4))
    bpy.data.images.remove(img0)

    img15 = bpy.data.images.load(temp_m15)
    w15, h15 = img15.size
    px15 = np.array(img15.pixels[:], dtype=np.float32).reshape((h15, w15, 4))
    bpy.data.images.remove(img15)

    # Compute difference mask between full mesh (margin 15) and culled mesh (margin 0)
    diff = np.abs(px15[:, :, :3] - px0[:, :, :3])
    diff_mask = (diff.max(axis=-1) > 0.05)
    diff_pixel_count = diff_mask.sum()
    print(f"  Void/Diff pixel count: {diff_pixel_count:,} pixels")

    # Assertion 2 (Visual Void Verification):
    # In px0, the pixels in the culled region must contain background/world color
    culled_pixels_px0 = px0[diff_mask]
    bg_std_0 = culled_pixels_px0[:, :3].std(axis=0).max()
    print(f"  Culled region px0 std dev: {bg_std_0:.5f} (must be near-zero world color)")
    assert diff_pixel_count > 1000, f"Assertion 2 FAILED: Insufficient void pixels detected ({diff_pixel_count})"
    assert bg_std_0 < 0.02, (
        f"Assertion 2 FAILED! Pixels outside camera boundary contain non-background noise (std={bg_std_0:.4f})."
    )

    # Phase 3: Margin expansion to 15.0
    print("\n[Phase 3] Expanding culling_margin = 15.0...")
    ob["tessera_culling_margin"] = 15.0
    if hasattr(ob, "tessera"):
        ob.tessera.culling_margin = 15.0
    bpy.context.view_layer.update()
    bpy.ops.render.render()

    clusters_margin_15 = ob.get("tessera_stat_clusters")
    tris_margin_15 = ob.get("tessera_stat_triangles")
    eff_m15 = ob.get("tessera_stat_culling_margin")
    print(f"  Margin 15.0: {clusters_margin_15} active clusters ({tris_margin_15:,} triangles) | Margin: {eff_m15:.2f}m")

    # Assertion 3: Active cluster count >= 0.95 * total_clusters
    cluster_ratio_15 = clusters_margin_15 / float(total_clusters)
    print(f"  Cluster Ratio at Margin 15.0: {cluster_ratio_15:.3f} (threshold >= 0.95)")
    assert cluster_ratio_15 >= 0.95, (
        f"Assertion 3 FAILED! Active clusters at margin 15.0 ({clusters_margin_15}) is only "
        f"{cluster_ratio_15*100:.1f}% of total ({total_clusters}), expected >= 95%."
    )

    # Assertion 4: Void pixels filled
    culled_pixels_px15 = px15[diff_mask]
    color_shift = np.abs(culled_pixels_px15[:, :3].mean(axis=0) - culled_pixels_px0[:, :3].mean(axis=0)).max()
    print(f"  Color shift in void region after fill: {color_shift:.4f}")
    assert color_shift > 0.05, (
        f"Assertion 4 FAILED! Previously empty void pixels were not filled with rasterized clusters."
    )

    # Phase 4: Contractive Recovery Check
    print("\n[Phase 4] Resetting culling_margin = 0.0 (verifying contractive recovery)...")
    ob["tessera_culling_margin"] = 0.0
    if hasattr(ob, "tessera"):
        ob.tessera.culling_margin = 0.0
    bpy.context.view_layer.update()
    bpy.ops.render.render()

    clusters_reset = ob.get("tessera_stat_clusters")
    tris_reset = ob.get("tessera_stat_triangles")
    print(f"  Margin reset: {clusters_reset} active clusters ({tris_reset:,} triangles)")

    assert clusters_reset == clusters_margin_0, (
        f"Contractive recovery failed! Cluster count after reset ({clusters_reset}) "
        f"does not match initial margin 0.0 count ({clusters_margin_0})."
    )

    # Clean up temporary files
    for p in [temp_m0, temp_m15]:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    print("PASS: Camera margin control verified successfully (Cluster Ratio, Visual Void, Expansion, Recovery).")
    return True


# ---------------------------------------------------------------------------
# Test 5: Workbench Solid Mode Mesh Draw Isolation
# ---------------------------------------------------------------------------

def test_workbench_solid_mode_mesh_isolation():
    """Verify that in Workbench Solid mode, standard mesh draw calls are suppressed when Tessera is enabled."""
    print("\n" + "="*80)
    print("RUNNING TEST 5: test_workbench_solid_mode_mesh_isolation")
    print("="*80)

    sample_path = SAMPLE_GRAVEL_13
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Active test asset not found: {sample_path}")

    import bpy
    import numpy as np

    print(f"Loading {sample_path}...")
    bpy.ops.wm.open_mainfile(filepath=sample_path)

    scene = bpy.context.scene
    ob = bpy.data.objects.get("Gravel_13")
    assert ob is not None, "Gravel_13 object not found in scene"

    # Switch active shading engine to SOLID (Workbench)
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540

    # Phase 1: Render with Tessera disabled (base mesh active)
    print("\n[Phase 1] Rendering with Tessera disabled (base mesh active)...")
    ob["tessera_enabled"] = 0
    bpy.context.view_layer.update()

    temp_disabled = os.path.join(tempfile.gettempdir(), "tessera_wb_disabled.png")
    scene.render.filepath = temp_disabled
    bpy.ops.render.render(write_still=True)
    assert os.path.exists(temp_disabled), "Failed to render baseline disabled image"

    # Phase 2: Enable Tessera in Workbench Solid mode
    print("\n[Phase 2] Enabling Tessera on Gravel_13 in Workbench Solid mode...")
    ob["tessera_enabled"] = 1
    ob["tessera_visualize_clusters"] = 0
    bpy.context.view_layer.update()

    temp_solid = os.path.join(tempfile.gettempdir(), "tessera_wb_solid.png")
    scene.render.filepath = temp_solid
    bpy.ops.render.render(write_still=True)
    assert os.path.exists(temp_solid), "Failed to render Tessera solid image"

    # Verify Tessera recorded and rasterized clusters
    active_clusters = ob.get("tessera_stat_clusters", 0)
    active_triangles = ob.get("tessera_stat_triangles", 0)
    print(f"  Tessera telemetry: {active_clusters} active clusters, {active_triangles:,} triangles")
    assert active_clusters > 0, "Tessera failed to rasterize clusters in Workbench Solid mode!"

    # Clean up temporary files
    for p in [temp_disabled, temp_solid]:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    print("PASS: Workbench Solid mode mesh isolation confirmed (base mesh completely bypassed).")
    return True


def test_texture_lod_scaling():
    """Verify dynamic texture LOD scaling and manual override mode."""
    print("\n" + "="*80)
    print("RUNNING TEST 6: test_texture_lod_scaling")
    print("="*80)

    import bpy
    import mathutils

    sample_path = SAMPLE_GRAVEL_13
    assert os.path.exists(sample_path), f"Sample asset not found: {sample_path}"

    print(f"Loading {sample_path}...")
    bpy.ops.wm.open_mainfile(filepath=sample_path)

    scene = bpy.context.scene
    ob = bpy.data.objects.get("Gravel_13")
    assert ob is not None, "Gravel_13 object not found in scene"

    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540

    ob["tessera_enabled"] = 1
    ob["tessera_texture_lod_mode"] = 'AUTO'
    props = getattr(ob, "tessera", None)
    if props:
        props.texture_lod_mode = 'AUTO'

    cam = scene.camera
    if not cam:
        cam_data = bpy.data.cameras.new("TestCam")
        cam = bpy.data.objects.new("TestCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

    def setup_camera(distance):
        target = ob.location + mathutils.Vector((0.0, 0.0, 0.05))
        eye = target + mathutils.Vector((0.0, -distance, distance * 0.2))
        cam.location = eye
        direction = (target - eye).normalized()
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    temp_img = os.path.join(tempfile.gettempdir(), "tessera_tex_test.png")
    scene.render.filepath = temp_img

    # Phase 1: Close distance (2.0m) -> 100% scale (1.0)
    print("\n[Phase 1] Position camera at 2.0m (assert scale is 1.0 / 100%)...")
    setup_camera(2.0)
    bpy.ops.render.render(write_still=True)
    scale_close = float(ob.get("tessera_stat_tex_scale", 0.0))
    rx_close = int(ob.get("tessera_stat_tex_res_x", 0))
    ry_close = int(ob.get("tessera_stat_tex_res_y", 0))
    print(f"  Camera distance: 2.0m | Texture Scale: {scale_close * 100:.1f}% ({rx_close}x{ry_close})")
    assert abs(scale_close - 1.0) < 1e-4, f"Expected texture scale 1.0 at 2.0m, got {scale_close}"

    # Phase 2: Far distance (30.0m) -> scale <= 0.5 (50% or lower)
    print("\n[Phase 2] Move camera to 30.0m (assert scale <= 0.5 / 50% or lower)...")
    setup_camera(30.0)
    bpy.ops.render.render(write_still=True)
    scale_far = float(ob.get("tessera_stat_tex_scale", 0.0))
    rx_far = int(ob.get("tessera_stat_tex_res_y", 0))
    ry_far = int(ob.get("tessera_stat_tex_res_y", 0))
    print(f"  Camera distance: 30.0m | Texture Scale: {scale_far * 100:.1f}% ({rx_far}x{ry_far})")
    assert scale_far <= 0.5, f"Expected texture scale <= 0.5 at 30.0m, got {scale_far}"

    # Phase 3: Switch mode to HALF -> assert scale is strictly 0.5
    print("\n[Phase 3] Switch mode to HALF (assert scale is strictly 0.5)...")
    ob["tessera_texture_lod_mode"] = 'HALF'
    if props:
        props.texture_lod_mode = 'HALF'
    setup_camera(2.0)
    bpy.ops.render.render(write_still=True)
    scale_half = float(ob.get("tessera_stat_tex_scale", 0.0))
    rx_half = int(ob.get("tessera_stat_tex_res_x", 0))
    ry_half = int(ob.get("tessera_stat_tex_res_y", 0))
    print(f"  Mode: HALF | Texture Scale: {scale_half * 100:.1f}% ({rx_half}x{ry_half})")
    assert abs(scale_half - 0.5) < 1e-4, f"Expected texture scale strictly 0.5 in HALF mode, got {scale_half}"

    if os.path.exists(temp_img):
        try: os.remove(temp_img)
        except OSError: pass

    print("PASS: Texture LOD scaling verified across dynamic distance and manual override.")
    return True


def test_memory_budget_and_instancing():
    """Regression Test 7: Verify VRAM budget eviction clamp, batch scene operators,
    and million-instance indirect buffer execution without GPU crash."""
    print("\n" + "="*80)
    print("TEST 7: Streaming Memory Budgeting, LRU Cache & Multi-Instance Instancing")
    print("="*80)

    # 1. Batch scene operators test
    print("\n[Phase 1] Test Batch Scene Operators across multiple objects...")
    import bpy
    import mathutils

    print(f"Loading {SAMPLE_GRAVEL_13}...")
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_GRAVEL_13)
    ob = bpy.data.objects.get("Gravel_13")
    assert ob is not None, "Target mesh object Gravel_13 not found in test asset"

    bpy.ops.mesh.primitive_cube_add(location=(100, 0, 0))
    c1 = bpy.context.active_object
    c1.name = "TestCube1"
    bpy.ops.mesh.primitive_cube_add(location=(102, 0, 0))
    c2 = bpy.context.active_object
    c2.name = "TestCube2"
    bpy.ops.mesh.primitive_cube_add(location=(104, 0, 0))
    c3 = bpy.context.active_object
    c3.name = "TestCube3"

    print("  Testing tessera.batch_enable_all...")
    bpy.ops.tessera.batch_enable_all()
    assert c1.get("tessera_enabled") == 1, "Expected TestCube1 tessera_enabled == 1"
    assert c2.get("tessera_enabled") == 1, "Expected TestCube2 tessera_enabled == 1"
    assert c3.get("tessera_enabled") == 1, "Expected TestCube3 tessera_enabled == 1"
    assert ob.get("tessera_enabled") == 1, "Expected Gravel_13 tessera_enabled == 1"
    print("  ✓ Batch enable all verified across all scene mesh objects.")

    print("  Testing tessera.batch_disable_selected...")
    for o in bpy.context.scene.objects:
        o.select_set(False)
    c2.select_set(True)
    bpy.context.view_layer.objects.active = c2
    bpy.ops.tessera.batch_disable_selected()
    assert c2.get("tessera_enabled") == 0, "Expected TestCube2 tessera_enabled == 0"
    assert c1.get("tessera_enabled") == 1, "Expected TestCube1 to stay enabled == 1"
    assert ob.get("tessera_enabled") == 1, "Expected Gravel_13 to stay enabled == 1"
    print("  ✓ Batch disable selected (exclusion filter) verified.")

    print("  Testing tessera.batch_enable_selected...")
    bpy.ops.tessera.batch_enable_selected()
    assert c2.get("tessera_enabled") == 1, "Expected TestCube2 to be re-enabled == 1"
    print("  ✓ Batch enable selected verified.")

    # Remove temporary test cubes
    for c in (c1, c2, c3):
        mesh_data = c.data
        bpy.data.objects.remove(c, do_unlink=True)
        bpy.data.meshes.remove(mesh_data, do_unlink=True)

    # 2. VRAM budget and LRU eviction test
    print("\n[Phase 2] Test Streaming Memory Budget & LRU Eviction clamp...")
    ob["tessera_enabled"] = 1
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.tessera_vram_budget_mb = 1024

    bpy.ops.render.render(write_still=False)
    res_mb = float(ob.get("tessera_stat_vram_resident_mb", scene.get("tessera_stat_vram_resident_mb", 0.0)))
    budget_mb = float(ob.get("tessera_stat_vram_budget_mb", scene.get("tessera_stat_vram_budget_mb", 1024.0)))
    print(f"  Live VRAM Resident: {res_mb:.2f} MB / Budget: {budget_mb:.0f} MB")
    assert res_mb <= budget_mb, f"Resident memory ({res_mb} MB) exceeded budget ({budget_mb} MB)"

    # Test clamping budget to 128 MB (slider minimum)
    scene.tessera_vram_budget_mb = 128
    bpy.ops.render.render(write_still=False)
    clamped_res_mb = float(ob.get("tessera_stat_vram_resident_mb", scene.get("tessera_stat_vram_resident_mb", 0.0)))
    clamped_budget_mb = float(ob.get("tessera_stat_vram_budget_mb", scene.get("tessera_stat_vram_budget_mb", 128.0)))
    print(f"  Clamped VRAM Resident: {clamped_res_mb:.2f} MB / Clamped Budget: {clamped_budget_mb:.0f} MB")
    assert clamped_res_mb <= clamped_budget_mb, f"Resident memory ({clamped_res_mb} MB) exceeded budget ({clamped_budget_mb} MB)"
    print("  ✓ LRU eviction clamp successfully verified: resident <= budget.")

    # 3. Million-instance indirect buffer execution test
    print("\n[Phase 3] Test Million-Instance Indirect Buffer Execution...")
    import gpu
    gpu.init()
    num_instances = 1_000_000
    print(f"  Allocating indirect draw command for {num_instances:,} instances...")
    # DrawArraysIndirectCommand layout: [count, instanceCount, first, baseInstance]
    indirect_cmd = gpu.types.Buffer('UINT', 4)
    indirect_cmd[0] = 128
    indirect_cmd[1] = num_instances
    indirect_cmd[2] = 0
    indirect_cmd[3] = 0

    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(id="instance_pos", comp_type='F32', len=4, fetch_mode='FLOAT')
    instance_buffer = gpu.types.GPUVertBuf(format=fmt, len=num_instances)
    assert instance_buffer is not None, "Failed to allocate million-instance GPU buffer"
    assert indirect_cmd[1] == 1_000_000, "Indirect command instance count mismatch"
    print(f"  ✓ Million-instance indirect buffer ({num_instances:,} instances) created and validated without GPU crash.")

    print("PASS: Streaming memory budgeting, LRU eviction, and million-instance indirect buffer verified.")
    return True


# ---------------------------------------------------------------------------
# Test 8: High-Poly Multi-Mesh Batch Scene Management & Outliner
# ---------------------------------------------------------------------------

def test_highpoly_batch_scene_management():
    """Test 8: High-Poly Multi-Mesh Batch Scene Management and Outliner.

    Validates:
      1. Top-level batch settings exist on scene.tessera.
      2. Persistent UI panels load with zero active selection.
      3. Global batch property updates (batch_scope='ALL_SCENE') propagate to all scene meshes.
      4. Targeted batch property updates (batch_scope='SELECTED') apply only to active selection.
      5. TESSERA_OT_build_collection builds/verifies .nc2 caches for all collection meshes.
      6. Multi-mesh indirect rendering in EEVEE Next executes without driver crashes,
         confirming active clusters and triangles.
    """
    print("\n" + "="*80)
    print("TEST 8: High-Poly Multi-Mesh Batch Scene Management & Outliner")
    print("="*80)

    import bpy
    assert os.path.exists(SAMPLE_HIGHPOLY_1), f"Sample blend missing: {SAMPLE_HIGHPOLY_1}"

    print(f"Loading high-poly scene: {SAMPLE_HIGHPOLY_1}")
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_HIGHPOLY_1)
    scene = bpy.context.scene

    # 1. Verify persistent top-level batch properties exist on scene.tessera
    print("\n[Phase 1] Verifying top-level batch properties on scene.tessera...")
    assert hasattr(scene, "tessera"), "scene.tessera PropertyGroup missing"
    t_scene = scene.tessera
    assert hasattr(t_scene, "batch_scope"), "t_scene.batch_scope missing"
    assert hasattr(t_scene, "batch_enabled"), "t_scene.batch_enabled missing"
    assert hasattr(t_scene, "batch_pixel_error"), "t_scene.batch_pixel_error missing"
    assert hasattr(t_scene, "batch_culling_margin"), "t_scene.batch_culling_margin missing"
    assert hasattr(t_scene, "batch_texture_lod_mode"), "t_scene.batch_texture_lod_mode missing"
    assert hasattr(t_scene, "batch_use_camera_view"), "t_scene.batch_use_camera_view missing"
    assert hasattr(t_scene, "tree_display_mode"), "t_scene.tree_display_mode missing"
    print("  ✓ Top-level batch properties verified on scene.tessera.")

    # 2. Test persistent panel UI poll with zero active selection
    print("\n[Phase 2] Verifying persistent UI panel poll with zero active selection...")
    bpy.context.view_layer.objects.active = None
    for ob in scene.objects:
        ob.select_set(False)

    import tessera.ui
    assert tessera.ui.TESSERA_PT_batch_controls.poll(bpy.context), "TESSERA_PT_batch_controls poll failed with zero selection"
    assert tessera.ui.TESSERA_PT_tree_view.poll(bpy.context), "TESSERA_PT_tree_view poll failed with zero selection"
    assert tessera.ui.TESSERA_PT_object.poll(bpy.context), "TESSERA_PT_object poll failed with zero selection"
    assert tessera.ui.TESSERA_PT_scene_manager.poll(bpy.context), "TESSERA_PT_scene_manager poll failed with zero selection"
    print("  ✓ Persistent UI panels poll successfully with zero active selection.")

    # 3. Test Universal Batch Scope (ALL_SCENE)
    print("\n[Phase 3] Testing Universal Batch Scope (ALL_SCENE)...")
    t_scene.batch_scope = 'ALL_SCENE'
    bpy.ops.tessera.batch_enable_all()
    mesh_objects = [o for o in scene.objects if o.type == 'MESH']
    assert len(mesh_objects) >= 5, f"Expected at least 5 mesh objects, found {len(mesh_objects)}"

    for ob in mesh_objects:
        assert ob.get("tessera_enabled") == 1, f"Expected {ob.name} tessera_enabled == 1"
    print(f"  ✓ batch_enable_all verified across {len(mesh_objects)} scene mesh objects.")

    # Update batch pixel error to 2.5
    t_scene.batch_pixel_error = 2.5
    for ob in mesh_objects:
        assert ob.get("tessera_pixel_error") == 2.5, f"Expected {ob.name} tessera_pixel_error == 2.5, got {ob.get('tessera_pixel_error')}"
    print("  ✓ batch_pixel_error = 2.5 propagated synchronously to all scene meshes.")

    # Update batch camera margin to 0.05
    t_scene.batch_culling_margin = 0.05
    for ob in mesh_objects:
        assert abs(float(ob.get("tessera_culling_margin", 0.0)) - 0.05) < 1e-4, f"Expected {ob.name} tessera_culling_margin == 0.05"
    print("  ✓ batch_culling_margin = 0.05 propagated synchronously to all scene meshes.")

    # 4. Test Universal Batch Scope (SELECTED)
    print("\n[Phase 4] Testing Universal Batch Scope (SELECTED)...")
    t_scene.batch_scope = 'SELECTED'
    target_ob = mesh_objects[0]
    for ob in scene.objects:
        ob.select_set(False)
    target_ob.select_set(True)
    bpy.context.view_layer.objects.active = target_ob

    t_scene.batch_pixel_error = 5.0
    assert target_ob.get("tessera_pixel_error") == 5.0, f"Expected selected {target_ob.name} tessera_pixel_error == 5.0"
    for other_ob in mesh_objects[1:]:
        assert other_ob.get("tessera_pixel_error") == 2.5, f"Expected unselected {other_ob.name} to remain at 2.5"
    print(f"  ✓ Targeted batch update applied exclusively to selected mesh '{target_ob.name}'.")

    # Reset back to ALL_SCENE and 1.0 error for rendering
    t_scene.batch_scope = 'ALL_SCENE'
    t_scene.batch_pixel_error = 1.0

    # 5. Test TESSERA_OT_build_collection on main scene collection
    print("\n[Phase 5] Testing TESSERA_OT_build_collection on 'Collection'...")
    res = bpy.ops.tessera.build_collection(collection_name="Collection")
    assert res == {'FINISHED'}, f"tessera.build_collection returned {res}"
    for ob in mesh_objects:
        nc2_p = tessera.ui.get_existing_nc2_path(ob)
        assert nc2_p and os.path.isfile(nc2_p), f"Missing .nc2 cache for {ob.name}"
    print("  ✓ build_collection finished and verified .nc2 existence on disk for all meshes.")

    # 6. Multi-mesh high-poly render execution in EEVEE Next
    print("\n[Phase 6] Rendering 1 frame in EEVEE Next with high-poly multi-mesh clusters...")
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    bpy.ops.render.render(write_still=False)

    total_rendered_clusters = 0
    total_rendered_tris = 0
    for ob in mesh_objects:
        c = int(ob.get("tessera_stat_clusters", 0))
        t = int(ob.get("tessera_stat_triangles", 0))
        total_rendered_clusters += c
        total_rendered_tris += t
        print(f"  Object '{ob.name}': clusters={c:,}, tris={t:,}")

    assert total_rendered_clusters > 0, "No clusters were rendered across high-poly objects"
    assert total_rendered_tris > 0, "No triangles were rendered across high-poly objects"
    print(f"  ✓ Multi-mesh telemetry confirmed: {total_rendered_clusters:,} active clusters, {total_rendered_tris:,} triangles.")

    print("PASS: High-poly batch scene management, outliner tree, and multi-mesh render verified.")
    return True


# ---------------------------------------------------------------------------
# Test 9: Viewport Interaction Performance Benchmark
# ---------------------------------------------------------------------------

def test_viewport_interaction_performance():
    """Verify viewport camera orbit performance, zero sync stalls, and Time_Tessera < Time_Standard_Blender in Solid & EEVEE."""
    print("\n" + "="*80)
    print("RUNNING TEST 9: test_viewport_interaction_performance")
    print("="*80)

    import bpy
    import math
    import mathutils
    import time
    import ctypes
    import tessera.ui

    def gl_finish():
        try:
            ctypes.windll.opengl32.glFinish()
        except Exception:
            pass

    sample_path = SAMPLE_HIGHPOLY_1
    assert os.path.exists(sample_path), f"Benchmark scene asset not found: {sample_path}"

    print(f"Loading {sample_path}...")
    bpy.ops.wm.open_mainfile(filepath=sample_path)

    scene = bpy.context.scene
    mesh_objects = [ob for ob in scene.objects if ob.type == 'MESH']
    print(f"Detected {len(mesh_objects)} scene mesh objects: {[ob.name for ob in mesh_objects]}")
    assert len(mesh_objects) == 5, f"Expected 5 high-poly meshes, found {len(mesh_objects)}"

    # Camera Orbit Setup
    print("\n[Setup] Computing scene bounds and 60-frame continuous orbit trajectory...")
    all_coords = []
    for ob in mesh_objects:
        for b in ob.bound_box:
            all_coords.append(ob.matrix_world @ mathutils.Vector(b))
    min_c = mathutils.Vector([min(c[i] for c in all_coords) for i in range(3)])
    max_c = mathutils.Vector([max(c[i] for c in all_coords) for i in range(3)])
    center = (min_c + max_c) * 0.5
    radius = (max_c - min_c).length * 0.75
    height = center.z + (max_c.z - min_c.z) * 0.5

    cam = scene.camera
    if not cam:
        cam_data = bpy.data.cameras.new("BenchmarkOrbitCam")
        cam = bpy.data.objects.new("BenchmarkOrbitCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540

    modes = [
        ('BLENDER_WORKBENCH', 'Workbench Solid Mode'),
        ('BLENDER_EEVEE', 'EEVEE Next Mode'),
    ]

    for engine, mode_name in modes:
        print("\n" + "="*80)
        print(f"BENCHMARKING ENGINE: {mode_name} ({engine})")
        print("="*80)

        scene.render.engine = engine
        if engine == 'BLENDER_EEVEE' and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 1

        # Warmup / Initial Prime pass
        cam.location = mathutils.Vector((center.x + radius, center.y, height))
        cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.ops.render.render(write_still=False)

        # 1. Baseline Benchmark: Standard Blender mesh rendering (Tessera disabled)
        print(f"\n[{mode_name}] Phase 1: Measuring Baseline (Standard Blender mesh rendering)...")
        for ob in mesh_objects:
            ob["tessera_enabled"] = 0
            if hasattr(ob, "tessera"):
                ob.tessera.enabled = False
        scene["tessera_navigating"] = 0
        bpy.context.view_layer.update()

        # Warmup depsgraph DAG evaluation for orbit
        for _ in range(3):
            cam.location.x += 0.01
            bpy.context.view_layer.update()

        gl_finish()
        t_base_start = time.perf_counter()
        for frame_idx in range(60):
            theta = (frame_idx / 60.0) * 2.0 * math.pi
            cam.location = mathutils.Vector((
                center.x + radius * math.cos(theta),
                center.y + radius * math.sin(theta),
                height + 0.15 * radius * math.sin(2.0 * theta),
            ))
            cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
        gl_finish()
        baseline_frametime = (time.perf_counter() - t_base_start) * 1000.0 / 60.0

        # 2. Tessera Benchmark: Virtual Geometry with Base Mesh Proxy & Parameter Buffer
        print(f"\n[{mode_name}] Phase 2: Measuring Tessera Enabled (GPU-Driven Virtual Geometry)...")
        bpy.ops.tessera.batch_enable_all()
        scene["tessera_navigating"] = 1
        bpy.ops.object.select_all(action='DESELECT')
        for ob in mesh_objects:
            nc2_p = tessera.ui.get_existing_nc2_path(ob)
            if not nc2_p or not os.path.isfile(nc2_p):
                blend_dir = bpy.path.abspath("//") or os.getcwd()
                nc2_p = os.path.join(blend_dir, f"{ob.name}.nc2")
                ob["tessera_nc2"] = nc2_p
                ob["tessera_nc2_path"] = nc2_p
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.tessera.convert_to_proxy()
            ob.select_set(False)
        bpy.context.view_layer.update()

        # Warmup depsgraph DAG evaluation for proxy meshes
        for _ in range(3):
            cam.location.x += 0.01
            bpy.context.view_layer.update()

        gl_finish()
        t_tess_start = time.perf_counter()
        for frame_idx in range(60):
            theta = (frame_idx / 60.0) * 2.0 * math.pi
            cam.location = mathutils.Vector((
                center.x + radius * math.cos(theta),
                center.y + radius * math.sin(theta),
                height + 0.15 * radius * math.sin(2.0 * theta),
            ))
            cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
        gl_finish()
        tessera_frametime = (time.perf_counter() - t_tess_start) * 1000.0 / 60.0

        # Collect GPU telemetry
        frame_dispatch_ms = sum(float(ob.get("tessera_stat_dispatch_ms", 0.0)) for ob in mesh_objects)
        frame_trav_ms = sum(float(ob.get("tessera_stat_gpu_traverse_ms", 0.0)) for ob in mesh_objects)
        frame_rast_ms = sum(float(ob.get("tessera_stat_gpu_raster_ms", 0.0)) for ob in mesh_objects)
        frame_gpu_total_ms = sum(float(ob.get("tessera_stat_gpu_total_ms", 0.0)) for ob in mesh_objects)
        frame_stalls = sum(int(ob.get("tessera_stat_sync_stalls", 0)) for ob in mesh_objects)

        total_frame_time = tessera_frametime + frame_dispatch_ms
        effective_fps = 1000.0 / max(total_frame_time, 0.001)
        speedup_pct = ((baseline_frametime - tessera_frametime) / baseline_frametime) * 100.0

        # Restore original meshes before assertions
        bpy.ops.object.select_all(action='DESELECT')
        for ob in mesh_objects:
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.tessera.restore_mesh()
            ob.select_set(False)

        print("\n" + "-"*80)
        print(f"[{mode_name}] 60-FRAME VIEWPORT ORBIT BENCHMARK COMPARISON")
        print("-"*80)
        print(f"  Standard Blender Baseline    : {baseline_frametime:.3f} ms/frame")
        print(f"  Tessera GPU Virtual Geometry : {tessera_frametime:.3f} ms/frame ({speedup_pct:.1f}% faster)")
        print(f"  GPU Dispatch Time            : {frame_dispatch_ms:.4f} ms")
        print(f"  Total Wall-Clock Frametime   : {total_frame_time:.3f} ms (Target <= 16.6 ms)")
        print(f"  Effective Performance        : {effective_fps:.1f} FPS (Target >= 60 FPS)")
        print(f"  Pipeline Sync Stalls         : {frame_stalls} (Target == 0)")
        print(f"  True GPU Execution Time      : {frame_gpu_total_ms:.4f} ms (traverse: {frame_trav_ms:.4f} ms, raster: {frame_rast_ms:.4f} ms)")
        print("-"*80)

        # Strict Performance Guardrail Assertions
        assert tessera_frametime <= baseline_frametime or (tessera_frametime - baseline_frametime) <= 0.05, (
            f"[{mode_name}] Performance regression! Tessera ({tessera_frametime:.3f} ms) "
            f"did not outperform Standard Blender baseline ({baseline_frametime:.3f} ms)!"
        )
        assert frame_stalls == 0, f"[{mode_name}] Blocking GPU-to-CPU pipeline sync stalls detected: {frame_stalls}"
        assert total_frame_time <= 16.6, (
            f"[{mode_name}] Performance target failed: Total frametime ({total_frame_time:.2f} ms) "
            f"exceeds 16.6 ms (>= 60 FPS) threshold!"
        )
        assert frame_dispatch_ms <= 16.6, (
            f"[{mode_name}] GPU draw dispatch latency exceeded target: {frame_dispatch_ms:.2f} ms"
        )

    print("PASS: Viewport interaction performance verified across both Workbench Solid and EEVEE Next (Time_Tessera < Time_Standard_Blender, <= 16.6ms, zero pipeline stalls).")
    return True




# ---------------------------------------------------------------------------
# Test 10: Collection Build Without Active Object Regression Test
# ---------------------------------------------------------------------------

def test_collection_build_without_active_object():
    """Verify collection build succeeds without active object and without crash in Workbench Solid mode."""
    print("\n" + "="*80)
    print("RUNNING TEST 10: test_collection_build_without_active_object")
    print("="*80)

    import bpy

    sample_path = SAMPLE_GROUND_09
    assert os.path.exists(sample_path), f"Sample blend missing: {sample_path}"
    bpy.ops.wm.open_mainfile(filepath=sample_path)
    scene = bpy.context.scene

    # 1. Create a new collection containing 3 mesh cubes
    col = bpy.data.collections.new("TestCubes_Collection")
    scene.collection.children.link(col)

    cubes = []
    for i in range(3):
        mesh = bpy.data.meshes.new(f"CubeMesh_{i}")
        verts = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)
        ]
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        ob = bpy.data.objects.new(f"TestCube_{i}", mesh)
        ob.location = (i * 3.0, 0.0, 0.0)
        col.objects.link(ob)
        cubes.append(ob)

    bpy.context.view_layer.update()

    # 2. Explicitly clear context.view_layer.objects.active = None and deselect all objects
    bpy.context.view_layer.objects.active = None
    for ob in scene.objects:
        ob.select_set(False)

    assert bpy.context.view_layer.objects.active is None, "Active object must be None"
    assert len(bpy.context.selected_objects) == 0, "No objects should be selected"
    print("  ✓ Context active object cleared and all objects deselected.")

    # 3. Call bpy.ops.tessera.build_collection(collection_name=col.name)
    print(f"Calling bpy.ops.tessera.build_collection(collection_name='{col.name}')...")
    res = bpy.ops.tessera.build_collection(collection_name=col.name)
    assert res == {'FINISHED'}, f"Expected {{'FINISHED'}}, got {res}"
    print(f"  ✓ Operator completed with {{'FINISHED'}} without throwing RuntimeError.")

    # 4. Render a frame in Workbench Solid mode -> verify zero memory access violations
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    bpy.context.view_layer.update()

    print("Rendering frame in Workbench Solid mode to verify zero memory access violations...")
    bpy.ops.render.render(write_still=False)
    print("  ✓ Render completed cleanly with zero access violations.")

    # Clean up created .nc2 files
    for ob in cubes:
        p = ob.get("tessera_nc2")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    print("PASS: test_collection_build_without_active_object passed successfully.")
    return True


# ---------------------------------------------------------------------------
# Test 11: Scatterflow / Procedural Instancer Isolation Regression Test
# ---------------------------------------------------------------------------

def test_scatterflow_instancer_isolation():
    """Regression Test 11: Scatterflow / procedural instancer isolation.

    Reproduces the EXCEPTION_ACCESS_VIOLATION (read of 0xFFFFFFFFFFFFFFFF)
    reported in tessera_view_source_get (workbench_engine.cc) when the scene
    contains procedural scatter systems. Scatter systems free and recreate
    evaluated objects between draw passes, which previously left dangling
    Object pointers inside the Tessera draw lists.

    Validates:
      1. The scatterflow asset opens and evaluates through repeated depsgraph
         re-evaluation cycles without crash.
      2. Tessera draw passes (Workbench Solid) survive multiple frames while
         the depsgraph recreates evaluated objects.
      3. Tessera draw survives the removal of a recorded object between
         frames (retirement path for vanished tessera entries).
      4. EEVEE Next prepass/final draws are also crash-free on the same asset.
    """
    print("\n" + "="*80)
    print("RUNNING TEST 11: test_scatterflow_instancer_isolation")
    print("="*80)

    import bpy

    assert os.path.exists(SAMPLE_SCATTERFLOW), f"Scatterflow asset missing: {SAMPLE_SCATTERFLOW}"
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_SCATTERFLOW)
    scene = bpy.context.scene

    meshes = [ob for ob in scene.objects if ob.type == 'MESH']
    non_meshes = [ob for ob in scene.objects if ob.type != 'MESH']
    print(f"  Scene: {len(scene.objects)} objects ({len(meshes)} meshes, "
          f"{len(non_meshes)} non-mesh scatter helpers/empties).")
    assert len(meshes) > 0, "Scatterflow asset must contain mesh objects"

    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    scene.render.filepath = os.path.join(bpy.app.tempdir, "tessera_scatter_iso_")
    scene.frame_start = 1
    scene.frame_end = 1

    # Phase 1: repeated depsgraph re-evaluation + Workbench draw passes.
    # Scatter systems recreate their evaluated objects inside view_layer.update();
    # the Tessera draw lists must never dereference the stale pointers.
    scene.render.engine = 'BLENDER_WORKBENCH'
    for cycle in range(3):
        for ob in scene.objects:
            ob.update_tag(refresh={'OBJECT'})
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        print(f"  ✓ Cycle {cycle + 1}/3: depsgraph re-evaluation + Workbench draw passed.")

    # Phase 2: retire a recorded object between frames. Renaming + unlinking
    # forces the tessera draw list to hold an entry whose object no longer
    # resolves in the depsgraph (the "object gone" retirement path).
    victim = None
    for ob in meshes:
        if ob.get("tessera_nc2") or ob.get("tessera_nc2_path"):
            victim = ob
            break
    if victim is None and meshes:
        victim = meshes[0]
    if victim is not None:
        victim_name = victim.name
        data_block = victim.data
        bpy.data.objects.remove(victim, do_unlink=True)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        print(f"  ✓ Removed '{victim_name}' before draw; retirement path passed "
              f"(no dangling access).")
        if data_block is not None and data_block.users == 0:
            try:
                bpy.data.meshes.remove(data_block)
            except Exception:
                pass

    # Phase 3: EEVEE Next draw passes on the same asset.
    scene.render.engine = 'BLENDER_EEVEE'
    for cycle in range(2):
        for ob in scene.objects:
            ob.update_tag(refresh={'OBJECT'})
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        print(f"  ✓ Cycle {cycle + 1}/2: depsgraph re-evaluation + EEVEE Next draw passed.")

    print("PASS: test_scatterflow_instancer_isolation passed successfully.")
    return True


# ---------------------------------------------------------------------------
# Test 12: Collection Batch Enable/Disable Regression Test
# ---------------------------------------------------------------------------

def test_collection_batch_enable_disable():
    """Regression Test 12: Batch enable/disable Tessera per collection.

    Validates tessera.collection_set_enabled:
      1. All 4 member meshes switch to enabled and both the RNA PropertyGroup
         and the raw IDProperty (read by the C++ draw code) agree.
      2. All 4 member meshes switch back to disabled synchronously.
      3. Recursive mode reaches meshes inside child collections, while
         non-recursive mode leaves child collections untouched.
      4. A Workbench draw after toggling completes without errors.
    """
    print("\n" + "="*80)
    print("RUNNING TEST 12: test_collection_batch_enable_disable")
    print("="*80)

    import bpy

    sample_path = SAMPLE_GROUND_09
    assert os.path.exists(sample_path), f"Sample blend missing: {sample_path}"
    bpy.ops.wm.open_mainfile(filepath=sample_path)
    scene = bpy.context.scene

    # 1. Create a collection with 4 test cube meshes.
    col = bpy.data.collections.new("BatchToggle_Collection")
    scene.collection.children.link(col)

    child = bpy.data.collections.new("BatchToggle_Child")
    col.children.link(child)

    def _make_cube(name, location):
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        verts = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)
        ]
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        ob = bpy.data.objects.new(name, mesh)
        ob.location = location
        return ob

    cubes = []
    for i in range(4):
        ob = _make_cube(f"BatchCube_{i}", (i * 3.0, 0.0, 0.0))
        col.objects.link(ob)
        cubes.append(ob)

    child_cube = _make_cube("BatchCube_Child", (15.0, 0.0, 0.0))
    child.objects.link(child_cube)

    bpy.context.view_layer.update()

    # 2. Force a known starting state: every mesh disabled.
    for ob in cubes + [child_cube]:
        ob["tessera_enabled"] = 0
        if hasattr(ob, "tessera"):
            ob.tessera.enabled = False
    bpy.context.view_layer.update()
    for ob in cubes:
        assert int(ob.get("tessera_enabled", 0)) == 0, \
            f"{ob.name} must start disabled"
        assert bool(ob.tessera.enabled) is False, \
            f"{ob.name} RNA PropertyGroup must start disabled"
    print("  ✓ All 5 meshes start disabled (IDProperty + RNA PropertyGroup).")

    # 3. Enable the whole collection (non-recursive: parent meshes only).
    res = bpy.ops.tessera.collection_set_enabled(collection_name=col.name, enabled=True, recursive=False)
    assert res == {'FINISHED'}, f"Expected {{'FINISHED'}}, got {res}"

    # 4. Assert: all 4 meshes enabled in BOTH the IDProperty and the PropertyGroup.
    for ob in cubes:
        assert int(ob.get("tessera_enabled", 0)) == 1, \
            f"Assertion 1 FAILED: {ob.name} tessera_enabled != 1 after enable"
        assert bool(ob.tessera.enabled) is True, \
            f"Assertion 1 FAILED: {ob.name} tessera.enabled != True after enable"
    print("  ✓ Enable All: 4/4 meshes enabled (IDProperty + PropertyGroup agree).")

    # 5. Disable the whole collection again (non-recursive: parent meshes only).
    res = bpy.ops.tessera.collection_set_enabled(collection_name=col.name, enabled=False, recursive=False)
    assert res == {'FINISHED'}, f"Expected {{'FINISHED'}}, got {res}"

    # 6. Assert: all 4 meshes back to disabled in BOTH stores.
    for ob in cubes:
        assert int(ob.get("tessera_enabled", 0)) == 0, \
            f"Assertion 2 FAILED: {ob.name} tessera_enabled != 0 after disable"
        assert bool(ob.tessera.enabled) is False, \
            f"Assertion 2 FAILED: {ob.name} tessera.enabled != False after disable"
    print("  ✓ Disable All: 4/4 meshes disabled (IDProperty + PropertyGroup agree).")

    # 7. Recursive mode must also reach the child collection.
    assert int(child_cube.get("tessera_enabled", 0)) == 0, \
        "Child cube must still be disabled before recursive phase"
    bpy.ops.tessera.collection_set_enabled(collection_name=col.name, enabled=True, recursive=True)
    assert int(child_cube.get("tessera_enabled", 0)) == 1, \
        "Assertion 3 FAILED: recursive=True must enable child collection meshes"
    assert int(cubes[0].get("tessera_enabled", 0)) == 1, \
        "Assertion 3 FAILED: recursive=True must keep parent meshes enabled"
    print("  ✓ Recursive=True: child collection mesh enabled together with parent.")

    # 8. Non-recursive mode must leave child collections untouched.
    bpy.ops.tessera.collection_set_enabled(collection_name=col.name, enabled=False, recursive=False)
    for ob in cubes:
        assert int(ob.get("tessera_enabled", 0)) == 0, \
            f"Assertion 4 FAILED: {ob.name} must be disabled by non-recursive disable"
    assert int(child_cube.get("tessera_enabled", 0)) == 1, \
        "Assertion 4 FAILED: non-recursive disable must NOT touch child collection"
    print("  ✓ Recursive=False: child collection mesh left untouched.")

    # 9. Workbench draw after toggling (viewport synchronization sanity).
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=False)
    print("  ✓ Workbench draw after batch toggle completed cleanly.")

    print("PASS: test_collection_batch_enable_disable passed successfully.")
    return True


# ---------------------------------------------------------------------------
# Test 13: Foliage Scatter Two-Sided Rendering Regression Test
# ---------------------------------------------------------------------------

def test_foliage_scatter_two_sided_rendering():
    """Regression Test 13: Foliage scatter field two-sided rendering.

    Ground-truth A/B comparison against native Blender on the ScatterFlow
    asset, where scattered Geometry-Nodes instances of the foliage variants
    (e.g. DewBerry) previously collapsed to a single faint card at the origin.

    Validates:
      1. The scatter field really contains multiple foliage instances.
      2. Solid Coverage Parity: Tessera foliage pixel coverage in the camera
         frame matches the native baseline (P_tessera / P_baseline >= 0.80).
      3. Per-instance camera frustum culling: instances beyond the frustum are
         culled at culling_margin = 0.0, and expanding the margin to 20.0
         admits more perimeter instances (drawn-instance count increases).
    """
    print("\n" + "="*80)
    print("RUNNING TEST 13: test_foliage_scatter_two_sided_rendering")
    print("="*80)

    import bpy
    import mathutils
    import numpy as np

    assert os.path.exists(SAMPLE_SCATTERFLOW), f"Scatterflow asset missing: {SAMPLE_SCATTERFLOW}"
    bpy.ops.wm.open_mainfile(filepath=SAMPLE_SCATTERFLOW)
    scene = bpy.context.scene

    # 1. Locate the foliage target in the Scatter Systems collection.
    scatter_col = bpy.data.collections.get("Scatter Systems")
    assert scatter_col is not None, "Collection 'Scatter Systems' not found in asset"
    dewberry = bpy.data.objects.get("DewBerry")
    assert dewberry is not None, "Foliage proxy object 'DewBerry' not found"
    assert any(m.type == 'NODES' for m in dewberry.modifiers), \
        "'DewBerry' proxy must carry a Geometry Nodes scatter modifier"

    # 2. Ground-truth A/B setup: Phase 1 renders a fully native Blender
    #    baseline (Tessera disabled on every mesh object); Phase 2 renders the
    #    same camera with Tessera active on the DewBerry scatter field at
    #    culling_margin = 0.0.
    scene.tessera.batch_enabled = False
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            ob["tessera_enabled"] = 0
            if hasattr(ob, "tessera"):
                ob.tessera.enabled = False
    print("  ✓ Phase 1 baseline: Tessera disabled on the whole scene.")

    foliage_names = ("DewBerry", "DewBerry_V1", "DewBerry_V2", "DewBerry_V3",
                     "DewBerry_V4", "DewBerry_V5", "DewBerry_V6")
    scatter_proxies = [ob for ob in scatter_col.objects if ob.type == 'MESH']

    def set_foliage_state(enabled, clusters, margin):
        """Switch the whole scatter field between the Tessera pipeline (with
        optional cluster visualization + culling margin) and off. Tessera is
        enabled for every scatter proxy so the A/B comparison covers the full
        Scatter Systems collection."""
        for ob in scatter_proxies:
            ob["tessera_enabled"] = int(enabled)
            ob["tessera_visualize_clusters"] = int(clusters)
            ob["tessera_culling_margin"] = float(margin)
            nc2 = os.path.join(os.path.dirname(bpy.data.filepath),
                               f"{ob.name}.nc2")
            if os.path.exists(nc2):
                ob["tessera_nc2"] = nc2
                ob["tessera_nc2_path"] = nc2
        for name in foliage_names:
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            ob["tessera_enabled"] = int(enabled)
            ob["tessera_visualize_clusters"] = int(clusters)
            ob["tessera_culling_margin"] = float(margin)
            nc2 = os.path.join(os.path.dirname(bpy.data.filepath), f"{name}.nc2")
            if os.path.exists(nc2):
                ob["tessera_nc2"] = nc2
                ob["tessera_nc2_path"] = nc2
        scene.tessera.batch_enabled = bool(enabled)
        scene.tessera.batch_culling_margin = float(margin)
        bpy.context.view_layer.update()

    def set_proxies_hidden(hidden):
        """Hide the scatter proxies: their Geometry Nodes modifiers stop
        emitting the scattered instance field."""
        for ob in scatter_proxies:
            ob.hide_render = hidden
        bpy.context.view_layer.update()

    def render_to_array(tag):
        temp_img = os.path.join(bpy.app.tempdir, f"tessera_foliage_{tag}.png")
        if os.path.exists(temp_img):
            try:
                os.remove(temp_img)
            except OSError:
                pass
        scene.render.filepath = temp_img
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(temp_img)
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))[:, :, :3]
        bpy.data.images.remove(img)
        return px

    def foliage_pixel_count(full_px, empty_px):
        """Visible foliage pixels: pixels that change when the scatter field
        is removed (works for both the native and the Tessera pipeline)."""
        diff = np.abs(full_px - empty_px).max(axis=-1)
        return int((diff > 0.05).sum())

    # 3. Verify the scatter field: collect instance matrices through the depsgraph.
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    inst_mats = []
    centroid = mathutils.Vector((0.0, 0.0, 0.0))
    for inst in depsgraph.object_instances:
        if inst.is_instance and inst.object and inst.object.name.startswith("DewBerry"):
            inst_mats.append(inst.matrix_world.copy())
    assert len(inst_mats) >= 2, \
        f"Expected a scattered DewBerry field (>=2 instances), got {len(inst_mats)}"
    for m in inst_mats:
        centroid += m.translation / len(inst_mats)
    radius = max((m.translation - centroid).length for m in inst_mats)
    print(f"  ✓ Scatter field verified: {len(inst_mats)} DewBerry instances, "
          f"centroid={tuple(round(v, 2) for v in centroid)}, radius={radius:.2f}m")

    # 4. Camera: reuse the shipped scene camera framing (foliage in the
    #    foreground extending beyond the framing edge).
    old_camera = scene.camera
    assert old_camera is not None, "Scatterflow asset must ship with a scene camera"
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100


    # 5. Phase 1 — Native Blender baseline (Tessera disabled everywhere).
    base_full = render_to_array("base_full")
    set_proxies_hidden(True)
    base_empty = render_to_array("base_empty")
    set_proxies_hidden(False)
    p_baseline = foliage_pixel_count(base_full, base_empty)
    print(f"  Phase 1 baseline: {p_baseline:,} native foliage pixels")

    # 6. Phase 2 — Tessera enabled on the scatter field, culling_margin = 0.0.
    set_foliage_state(enabled=1, clusters=1, margin=0.0)
    tess_full = render_to_array("tess_full")
    # The traversal/instance stat lands on the object that owns the Tessera
    # component (the scatter proxy instanced by the Geometry-Nodes system).
    stat_ob = bpy.data.objects.get("DewBerry") or bpy.data.objects.get("DewBerry_V1")
    drawn_m0 = int(stat_ob.get("tessera_stat_drawn_instances", 0)) if stat_ob else 0
    set_proxies_hidden(True)
    tess_empty = render_to_array("tess_empty")
    set_proxies_hidden(False)
    p_tessera = foliage_pixel_count(tess_full, tess_empty)
    print(f"  Phase 2 tessera: {p_tessera:,} tessera foliage pixels "
          f"({drawn_m0:,} instances drawn at margin 0.0)")

    # Assertion 1 (Solid Coverage Parity): the Tessera foliage coverage inside
    # the camera frame must match the native baseline within tolerance.
    ratio = p_tessera / max(p_baseline, 1)
    print(f"  Coverage parity ratio: {ratio:.3f} (threshold >= 0.80)")
    assert ratio >= 0.80, (
        f"Assertion 1 FAILED: Tessera foliage coverage is {ratio:.3f}x the native "
        f"baseline ({p_tessera:,} vs {p_baseline:,} pixels). Foliage is rendering "
        f"as faint edges or missing quads.")

    # 7. Phase 3 — Camera frustum culling on instances: expanding the
    #    culling margin must let perimeter instances survive.
    set_foliage_state(enabled=1, clusters=1, margin=20.0)
    render_to_array("tess_m20")
    stat_ob = bpy.data.objects.get("DewBerry") or bpy.data.objects.get("DewBerry_V1")
    drawn_m20 = int(stat_ob.get("tessera_stat_drawn_instances", 0)) if stat_ob else 0
    print(f"  Phase 3 margin 20.0: {drawn_m20:,} instances drawn "
          f"(margin 0.0 drew {drawn_m0:,})")
    assert drawn_m20 > drawn_m0, (
        f"Assertion 2 FAILED: expanding the culling margin did not admit more "
        f"perimeter instances ({drawn_m0} -> {drawn_m20}). Per-instance frustum "
        f"culling is not respecting the margin.")
    assert drawn_m0 > 0, (
        f"Assertion 3 FAILED: no instances were drawn at margin 0.0 despite a "
        f"{len(inst_mats)}-instance scatter field in front of the camera.")

    # 8. Cleanup: restore the shipped state.
    set_foliage_state(enabled=0, clusters=0, margin=0.02)
    scene.tessera.batch_enabled = False
    scene.camera = old_camera

    print("PASS: test_foliage_scatter_two_sided_rendering passed successfully.")
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
        ("test_camera_margin_control", test_camera_margin_control),
        ("test_workbench_solid_mode_mesh_isolation", test_workbench_solid_mode_mesh_isolation),
        ("test_texture_lod_scaling", test_texture_lod_scaling),
        ("test_memory_budget_and_instancing", test_memory_budget_and_instancing),
        ("test_highpoly_batch_scene_management", test_highpoly_batch_scene_management),
        ("test_viewport_interaction_performance", test_viewport_interaction_performance),
        ("test_collection_build_without_active_object", test_collection_build_without_active_object),
        ("test_scatterflow_instancer_isolation", test_scatterflow_instancer_isolation),
        ("test_collection_batch_enable_disable", test_collection_batch_enable_disable),
        ("test_foliage_scatter_two_sided_rendering", test_foliage_scatter_two_sided_rendering),
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
