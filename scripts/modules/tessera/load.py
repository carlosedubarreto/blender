"""Loading / saving Tessera cluster data."""

import bpy
from bpy.props import StringProperty

from .build import TesseraData, serialize_to_memory, deserialize_from_memory

# Module-level cache of the most recently loaded Tessera data. Arbitrary
# Python objects cannot be stored on ID types (WindowManager), so this is
# the pragmatic way to pass data to the render module.
_loaded_data = None


def get_loaded_data():
    """Return the TesseraData loaded by the last tessera.load execution."""
    return _loaded_data


class TesseraLoad(bpy.types.Operator):
    """Load Tessera cluster data from a .nc2 file into a Python object."""
    bl_idname = "tessera.load"
    bl_label = "Load Tessera Clusters"
    bl_options = {"REGISTER", "UNDO"}

    path: StringProperty(
        name="Input .nc2 path",
        description="Path of the Tessera .nc2 file to load",
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
        # TesseraData would be loaded from.
        global _loaded_data
        _loaded_data = data

        self.report({"INFO"}, f"Loaded Tessera data from {self.path}")
        return {"FINISHED"}


class TesseraClear(bpy.types.Operator):
    """Remove Tessera data from the active object (falls back to regular mesh rendering)"""
    bl_idname = "tessera.clear"
    bl_label = "Clear Tessera Data"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and "tessera_nc2" in ob

    def execute(self, context):
        ob = context.active_object

        # Reset the RNA properties first: their update callbacks write the
        # IDProperties back, so the IDProperty removal must come last.
        if hasattr(bpy.types.Object, "tessera"):
            props = ob.tessera
            props.enabled = True
            props.pixel_error = 1.0
            props.visualize_clusters = False
            props.nc2_path = ""

        for key in ("tessera_nc2",
                    "tessera_enabled",
                    "tessera_pixel_error",
                    "tessera_visualize_clusters"):
            if key in ob:
                del ob[key]

        ob.update_tag()
        if context.area is not None:
            context.area.tag_redraw()

        self.report({"INFO"}, "Tessera data cleared")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(TesseraLoad)
    bpy.utils.register_class(TesseraClear)


def unregister():
    bpy.utils.unregister_class(TesseraClear)
    bpy.utils.unregister_class(TesseraLoad)
