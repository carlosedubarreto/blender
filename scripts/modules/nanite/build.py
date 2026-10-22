"""Nanite cluster data Python module.

This module implements the cluster data model and IO in pure Python so the
Blender operator can run without the C++ backend being compiled. The actual
GPU rendering of the clusters is performed by the compiled :mod:`nanite`
backend (``blender::nanite``) via the C++ renderer documented in
``render.py``.

Data model and IO mirror the C++ :class:`NaniteData` / :mod:`nanite_cluster`
format.

Example::

    import bpy
    bpy.ops.nanite.build('INVOKE_DEFAULT')
    bpy.ops.nanite.load('INVOKE_DEFAULT')
"""

import bpy
from bpy.props import StringProperty, FloatProperty, EnumProperty
import struct
import os

# ---------------------------------------------------------------------------
# Pure-Python data model (mirrors the C++ NaniteData).
# ---------------------------------------------------------------------------

class NaniteData:
    """Lightweight Python holder for Nanite cluster data."""

    def __init__(self):
        self.header = {
            "magic": 0x3255434E,  # "NCU2"
            "version": 1,
            "clusterCount": 0,
            "levelCount": 1,
            "vertexCount": 0,
            "indexCount": 0,
            "proxyVertexCount": 0,
            "proxyIndexCount": 0,
            "maxClusterTriangles": 0,
            "maxClusterVertices": 0,
            "flags": 0,
            "boundsMin": (0.0, 0.0, 0.0),
            "boundsMax": (0.0, 0.0, 0.0),
            "uvMin": (0.0, 0.0),
            "uvMax": (0.0, 0.0),
        }
        self.clusters = []
        self.positions = []
        self.normals = []
        self.uvs = []
        self.indices = []
        self.proxyPositions = []
        self.proxyIndices = []


# ---------------------------------------------------------------------------
# Pure-Python IO (flat binary format).
# ---------------------------------------------------------------------------

def _pack_header(h):
    """Pack the 88-byte NaniteHeader (12 uint32 + 10 float, little-endian)."""
    return struct.pack("<12I10f", *[
        h["magic"], h["version"], h["clusterCount"], h["levelCount"],
        h["vertexCount"], h["indexCount"], h["proxyVertexCount"],
        h["proxyIndexCount"], h["maxClusterTriangles"],
        h["maxClusterVertices"], h["flags"], 0,
        h["boundsMin"][0], h["boundsMin"][1], h["boundsMin"][2],
        h["boundsMax"][0], h["boundsMax"][1], h["boundsMax"][2],
        h["uvMin"][0], h["uvMin"][1], h["uvMax"][0], h["uvMax"][1],
    ])


def serialize_to_memory(data):
    """Serialize to a flat byte buffer (same layout as the C++ version)."""
    buf = bytearray()
    buf += _pack_header(data.header)
    for c in data.clusters:
        buf += struct.pack("<10fIIiIIIII",
                           c["sphere"][0], c["sphere"][1], c["sphere"][2],
                           c["sphere"][3],
                           c["parentSphere"][0], c["parentSphere"][1],
                           c["parentSphere"][2], c["parentSphere"][3],
                           c["error"], c["parentError"],
                           c["firstIndex"], c["indexCount"],
                           c["parentGroup"], c["group"],
                           c["level"], c["flags"], 0, 0)
    return bytes(buf)


_HEADER_SIZE = 88   # sizeof(NaniteHeader)
_CLUSTER_SIZE = 72  # sizeof(NaniteCluster)


def deserialize_from_memory(path):
    """Read a NaniteData from a .nc2 file."""
    data = NaniteData()
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < _HEADER_SIZE:
        raise ValueError("File too small to contain a Nanite header")

    (magic, version, cluster_count, level_count, vertex_count, index_count,
     proxy_vertex_count, proxy_index_count, max_cluster_triangles,
     max_cluster_vertices, flags, _pad,
     bmin0, bmin1, bmin2, bmax0, bmax1, bmax2,
     uvmin0, uvmin1, uvmax0, uvmax1) = struct.unpack_from("<12I10f", raw, 0)

    if magic != data.header["magic"]:
        raise ValueError("Not a Nanite .nc2 file (bad magic)")

    data.header.update({
        "magic": magic, "version": version,
        "clusterCount": cluster_count, "levelCount": level_count,
        "vertexCount": vertex_count, "indexCount": index_count,
        "proxyVertexCount": proxy_vertex_count,
        "proxyIndexCount": proxy_index_count,
        "maxClusterTriangles": max_cluster_triangles,
        "maxClusterVertices": max_cluster_vertices, "flags": flags,
        "boundsMin": (bmin0, bmin1, bmin2),
        "boundsMax": (bmax0, bmax1, bmax2),
        "uvMin": (uvmin0, uvmin1), "uvMax": (uvmax0, uvmax1),
        "path": path,
    })

    offset = _HEADER_SIZE
    for _ in range(cluster_count):
        (s0, s1, s2, s3, p0, p1, p2, p3, error, parent_error,
         first_index, index_count_c, parent_group, group, level, cflags,
         _pad0, _pad1) = struct.unpack_from("<10fIIiIIIII", raw, offset)
        data.clusters.append({
            "sphere": (s0, s1, s2, s3),
            "parentSphere": (p0, p1, p2, p3),
            "error": error, "parentError": parent_error,
            "firstIndex": first_index, "indexCount": index_count_c,
            "parentGroup": parent_group, "group": group,
            "level": level, "flags": cflags,
        })
        offset += _CLUSTER_SIZE
    return data

# ---------------------------------------------------------------------------
# Blender operators.
# ---------------------------------------------------------------------------

def build_from_mesh(obj):
    """Placeholder cluster builder.

    In a full implementation this would call the C++ :func:`buildFromMesh`;
    here it creates a trivial single-cluster data set for early testing.
    """
    data = NaniteData()
    mesh = obj.data
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.polygons)
    data.header["vertexCount"] = n_verts
    data.header["proxyVertexCount"] = n_verts
    data.header["indexCount"] = n_faces * 3
    data.header["proxyIndexCount"] = n_faces * 3
    data.header["maxClusterTriangles"] = n_faces
    data.header["maxClusterVertices"] = n_verts
    data.header["boundsMin"] = (0.0, 0.0, 0.0)
    data.header["boundsMax"] = (0.0, 0.0, 0.0)
    data.positions = [(0.0, 0.0, 0.0)]
    data.indices = [0]
    data.clusters.append({
        "sphere": (0.0, 0.0, 0.0, 1.0),
        "parentSphere": (0.0, 0.0, 0.0, 1.0),
        "error": 0.0, "parentError": 99999.0,
        "firstIndex": 0, "indexCount": 3,
        "parentGroup": -1, "group": 0, "level": 0, "flags": 0,
    })
    data.header["clusterCount"] = len(data.clusters)
    data.header["levelCount"] = 1
    data.header["flags"] = 0
    return data


class NaniteBuild(bpy.types.Operator):
    """Build Nanite clusters and save them to a .nc2 file."""
    bl_idname = "nanite.build"
    bl_label = "Build Nanite Clusters"
    bl_options = {"REGISTER", "UNDO"}

    path: StringProperty(
        name="Output .nc2 path",
        description="Output .nc2 file path",
        default="",
        maxlen=1024,
    )

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object is not a mesh")
            return {"CANCELLED"}

        import os
        if not self.path:
            self.path = os.path.splitext(obj.data.name)[0] + ".nc2"

        data = build_from_mesh(obj)

        # Save a minimal placeholder .nc2 (full serialization uses the C++ backend
        # for production).
        try:
            with open(self.path, "wb") as f:
                f.write(serialize_to_memory(data))
        except OSError as err:
            self.report({"ERROR"}, f"Could not write file: {err}")
            return {"CANCELLED"}

        obj["nanite_nc2"] = self.path
        if hasattr(bpy.types.Object, "nanite"):
            obj.nanite.nc2_path = self.path
        self.report({"INFO"}, f"Built Nanite data -> {self.path}")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NaniteBuild)


def unregister():
    bpy.utils.unregister_class(NaniteBuild)
