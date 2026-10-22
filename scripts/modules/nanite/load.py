"""Loading / saving Nanite cluster data."""

import bpy
from bpy.props import StringProperty

from .build import NaniteData, serialize_to_memory, deserialize_from_memory

# Module-level cache of the most recently loaded Nanite data. Arbitrary
# Python objects cannot be stored on ID types (WindowManager), so this is
# the pragmatic way to pass data to the render module.
_loaded_data = None


def get_loaded_data():
    """Return the NaniteData loaded by the last nanite.load execution."""
    return _loaded_data


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

        # The Python implementation stores the data in this module for later
        # use by the render backend. In a full build, this is where the C++
        # NaniteData would be loaded from.
        global _loaded_data
        _loaded_data = data

        self.report({"INFO"}, f"Loaded Nanite data from {self.path}")
        return {"FINISHED"}


class NaniteClear(bpy.types.Operator):
    """Remove Nanite data from the active object (falls back to regular mesh rendering)"""
    bl_idname = "nanite.clear"
    bl_label = "Clear Nanite Data"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and "nanite_nc2" in ob

    def execute(self, context):
        ob = context.active_object

        # Reset the RNA properties first: their update callbacks write the
        # IDProperties back, so the IDProperty removal must come last.
        if hasattr(bpy.types.Object, "nanite"):
            props = ob.nanite
            props.enabled = True
            props.pixel_error = 1.0
            props.visualize_clusters = False
            props.nc2_path = ""

        for key in ("nanite_nc2",
                    "nanite_enabled",
                    "nanite_pixel_error",
                    "nanite_visualize_clusters"):
            if key in ob:
                del ob[key]

        ob.update_tag()
        if context.area is not None:
            context.area.tag_redraw()

        self.report({"INFO"}, "Nanite data cleared")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(NaniteLoad)
    bpy.utils.register_class(NaniteClear)


def unregister():
    bpy.utils.unregister_class(NaniteClear)
    bpy.utils.unregister_class(NaniteLoad)
