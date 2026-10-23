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
        nanite_props = getattr(ob, "nanite", None)

        col = layout.column(align=True)
        col.operator("nanite.build", icon='MOD_BUILD')

        nc2_path = ob.get("nanite_nc2")
        if nc2_path:
            col.operator("nanite.clear", icon='X')

            box = layout.box()
            box.label(text="Nanite data built", icon='CHECKMARK')
            box.label(text=os.path.basename(nc2_path), icon='FILE')

            if nanite_props is not None:
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                col.prop(nanite_props, "enabled", text="Use Nanite")
                col.prop(nanite_props, "pixel_error", text="Pixel Error")
                col.prop(nanite_props, "view_source", text="View Source")
                col.prop(nanite_props, "visualize_clusters")
        else:
            if nanite_props is not None:
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                col.prop(nanite_props, "enabled", text="Use Nanite")

            box = layout.box()
            box.label(text="No Nanite data", icon='INFO')
            box.label(text="Build clusters to render with Nanite")

        # Statistics from the most recently loaded data (if any).
        from nanite import load as _load
        data = _load.get_loaded_data()
        if data is not None:
            header = data.header
            box = layout.box()
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