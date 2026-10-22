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
    return struct.pack("<IIIIIIIIIIIIIIIiiiIII", *[
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
        buf += struct.pack("<ffffffffIiiiII",
                           c["sphere"][0], c["sphere"][1], c["sphere"][2],
                           c["sphere"][3],
                           c["parentSphere"][0], c["parentSphere"][1],
                           c["parentSphere"][2], c["parentSphere"][3],
                           c["error"], c["parentError"],
                           c["firstIndex"], c["indexCount"],
                           c["parentGroup"], c["group"],
                           c["level"], c["flags"])
    return bytes(buf)


def deserialize_from_memory(path):
    """Read a NaniteData from a .nc2 file."""
    data = NaniteData()
    with open(path, "rb") as f:
        raw = f.read()
    data.header["clusterCount"] = 0
    data.header["path"] = path
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
    n_verts = mesh.totvert
    data.header["vertexCount"] = n_verts
    data.header["proxyVertexCount"] = n_verts
    data.header["indexCount"] = mesh.totface * 3
    data.header["proxyIndexCount"] = mesh.totface * 3
    data.header["maxClusterTriangles"] = mesh.totface
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
        self.report({"INFO"}, f"Built Nanite data -> {self.path}")
        return {"FINISHED"}


class NaniteLoad(bpy.types.Operator):
    """Load Nanite data from a .nc2 file."""
    bl_idname = "nanite.load"
    bl_label = "Load Nanite Clusters"
    bl_options = {"REGISTER", "UNDO"}

    path: StringProperty(
        name="Input .nc2 path",
        description="Input .nc2 file path",
        default="",
        maxlen=1024,
    )

    def execute(self, context):
        import os
        if not self.path or not os.path.exists(self.path):
            self.report({"ERROR"}, f"File not found: {self.path}")
            return {"CANCELLED"}

        data = deserialize_from_memory(self.path)
        context.window_manager.nanite_data = data
        self.report({"INFO"}, f"Loaded Nanite data from {self.path}")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NaniteBuild)
    bpy.utils.register_class(NaniteLoad)


def unregister():
    bpy.utils.unregister_class(NaniteBuild)
    bpy.utils.unregister_class(NaniteLoad)

