"""Tessera user interface.

Adds a *Tessera* panel to the 3D viewport sidebar (N-panel) exposing the
available Tessera tools: building cluster data from the active mesh, loading
previously saved ``.nc2`` files, and toggling cluster visualization.

The panel is automatically registered together with the rest of the
:mod:`tessera` module by :func:`tessera.register`.
"""

import os

import bpy


class TESSERA_PT_object(bpy.types.Panel):
    """Tessera options for the active mesh object."""

    bl_label = "Tessera"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        tessera_props = getattr(ob, "tessera", None)

        col = layout.column(align=True)
        col.operator("tessera.build", icon='MOD_BUILD')

        nc2_path = ob.get("tessera_nc2")
        if nc2_path:
            col.operator("tessera.clear", icon='X')

            box = layout.box()
            box.label(text="Tessera data built", icon='CHECKMARK')
            box.label(text=os.path.basename(nc2_path), icon='FILE')

            if tessera_props is not None:
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                col.prop(tessera_props, "enabled", text="Use Tessera")
                col.prop(tessera_props, "pixel_error", text="Pixel Error")
                col.prop(tessera_props, "view_source", text="View Source")
                col.prop(tessera_props, "visualize_clusters")
        else:
            if tessera_props is not None:
                col = layout.column(align=True)
                col.use_property_split = True
                col.use_property_decorate = False
                col.prop(tessera_props, "enabled", text="Use Tessera")

            box = layout.box()
            box.label(text="No Tessera data", icon='INFO')
            box.label(text="Build clusters to render with Tessera")

        # Statistics from the most recently loaded data (if any).
        from tessera import load as _load
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
    TESSERA_PT_object,
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)