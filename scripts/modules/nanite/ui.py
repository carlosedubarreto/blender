"""Nanite user interface.

Adds a *Nanite* panel to the 3D viewport sidebar (N-panel) exposing the
available Nanite tools: building cluster data from the active mesh, loading
previously saved ``.nc2`` files, and toggling cluster visualization.

The panel is automatically registered together with the rest of the
:mod:`nanite` module by :func:`nanite.register`.
"""

import os

import bpy


class NANITE_PT_object(bpy.types.Panel):
    """Nanite options for the active mesh object."""

    bl_label = "Nanite"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Nanite"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        ob = context.active_object

        col = layout.column(align=True)
        col.operator("nanite.build", icon='MOD_BUILD')
        col.operator("nanite.load", icon='FILE_FOLDER')

        nanite_props = getattr(ob, "nanite", None)
        if nanite_props is not None:
            col = layout.column(align=True)
            col.prop(nanite_props, "nc2_path", text="")
            col.prop(nanite_props, "visualize_clusters")

        # Status: custom property set by the build operator.
        nc2_path = ob.get("nanite_nc2")
        if nc2_path:
            box = layout.box()
            box.label(text="Nanite data built", icon='CHECKMARK')
            box.label(text=os.path.basename(nc2_path), icon='FILE')

            # Statistics from the most recently loaded data (if any).
            from nanite import load as _load
            data = _load.get_loaded_data()
            if data is not None:
                header = data.header
                box.label(
                    text="Clusters: {}, Verts: {}".format(
                        header["clusterCount"], header["vertexCount"]),
                    icon='MESH_DATA',
                )


classes = (
    NANITE_PT_object,
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)