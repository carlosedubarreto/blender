"""Loading / saving Nanite cluster data."""

import bpy
from bpy.props import StringProperty

from . import NaniteData, serialize_to_memory, deserialize_from_memory


class NaniteLoad(bpy.types.Operator):
    """Load Nanite cluster data from a .nc2 file into a Python object."""
    bl_idname = "nanite.load"
    bl_label = "Load Nanite Clusters"
    bl_options = {"REGISTER", "UNDO"}

    path: StringProperty(
        name="Input .nc2 path",
        description="Path of the Nanite .nc2 file to load",
        default="",
        maxlen=1024,
    )

    def execute(self, context):
        import os
        if not self.path or not os.path.exists(self.path):
            self.report({"ERROR"}, f"File not found: {self.path}")
            return {"CANCELLED"}

        data = deserialize_from_memory(self.path)
        if data.header["clusterCount"] == 0:
            self.report({"ERROR"}, "No clusters found in the .nc2 file")
            return {"CANCELLED"}

        # The Python implementation stores the file path for later use by the
        # render backend. In a full build, this is where the C++ NaniteData
        # would be loaded from.
        data.header["path"] = self.path

        context.window_manager.nanite_data = data

        self.report({"INFO"}, f"Loaded Nanite data from {self.path}")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NaniteLoad)


def unregister():
    bpy.utils.unregister_class(NaniteLoad)
