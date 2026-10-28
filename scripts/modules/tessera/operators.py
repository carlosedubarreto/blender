"""Tessera batch scene operators.

Provides batch operators to enable, disable, and build Tessera cluster data
across scene meshes and selected objects.
"""

import bpy
import os


class TESSERA_OT_batch_enable_all(bpy.types.Operator):
    """Enable Tessera for all valid mesh objects in the current scene"""
    bl_idname = "tessera.batch_enable_all"
    bl_label = "Enable for All Scene Objects"
    bl_description = "Registers all valid mesh objects in the scene into Tessera and generates missing caches"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.ui import get_existing_nc2_path
        from tessera.render import _tag_tessera_redraw

        count = 0
        for ob in context.scene.objects:
            if ob.type != 'MESH':
                continue
            ob["tessera_enabled"] = 1
            if getattr(ob, "tessera", None):
                ob.tessera.enabled = True
            nc2 = get_existing_nc2_path(ob)
            if nc2:
                ob["tessera_nc2"] = nc2
            count += 1
            _tag_tessera_redraw(ob, context)

        self.report({'INFO'}, f"Tessera enabled for {count} scene objects")
        return {'FINISHED'}


class TESSERA_OT_batch_enable_selected(bpy.types.Operator):
    """Enable Tessera for all selected mesh objects"""
    bl_idname = "tessera.batch_enable_selected"
    bl_label = "Enable for Selected Objects"
    bl_description = "Registers only active selection into Tessera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.ui import get_existing_nc2_path
        from tessera.render import _tag_tessera_redraw

        count = 0
        for ob in context.selected_objects:
            if ob.type != 'MESH':
                continue
            ob["tessera_enabled"] = 1
            if getattr(ob, "tessera", None):
                ob.tessera.enabled = True
            nc2 = get_existing_nc2_path(ob)
            if nc2:
                ob["tessera_nc2"] = nc2
            count += 1
            _tag_tessera_redraw(ob, context)

        self.report({'INFO'}, f"Tessera enabled for {count} selected objects")
        return {'FINISHED'}


class TESSERA_OT_batch_disable_selected(bpy.types.Operator):
    """Disable Tessera for selected objects (exclusion filter)"""
    bl_idname = "tessera.batch_disable_selected"
    bl_label = "Exclude Selected / Invert"
    bl_description = "Disables Tessera on selected objects while leaving the rest active"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.render import _tag_tessera_redraw

        count = 0
        for ob in context.selected_objects:
            ob["tessera_enabled"] = 0
            if getattr(ob, "tessera", None):
                ob.tessera.enabled = False
            count += 1
            _tag_tessera_redraw(ob, context)

        self.report({'INFO'}, f"Tessera disabled for {count} selected objects")
        return {'FINISHED'}


class TESSERA_OT_batch_build_all(bpy.types.Operator):
    """Batch build missing .nc2 cluster caches for all Tessera-enabled objects"""
    bl_idname = "tessera.batch_build_all"
    bl_label = "Batch Build All Missing .nc2"
    bl_description = "Headless multithreaded build trigger for all scene objects missing .nc2 caches"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.ui import get_existing_nc2_path

        missing = []
        for ob in context.scene.objects:
            if ob.type != 'MESH':
                continue
            if not ob.get("tessera_enabled", False):
                continue
            existing = get_existing_nc2_path(ob)
            if not existing:
                missing.append(ob)

        if not missing:
            self.report({'INFO'}, "All Tessera-enabled objects have valid .nc2 caches")
            return {'FINISHED'}

        built_count = 0
        blend_dir = bpy.path.abspath("//")
        if not blend_dir or not os.path.isdir(blend_dir):
            blend_dir = os.getcwd()

        for ob in missing:
            nc2_path = os.path.join(blend_dir, f"{ob.name}.nc2")
            ob["tessera_nc2"] = nc2_path
            ob["tessera_nc2_path"] = nc2_path
            try:
                orig_active = context.view_layer.objects.active
                context.view_layer.objects.active = ob
                bpy.ops.object.tessera_build(filepath=nc2_path)
                context.view_layer.objects.active = orig_active
                built_count += 1
            except Exception:
                pass

        self.report({'INFO'}, f"Triggered build for {built_count} missing .nc2 caches")
        return {'FINISHED'}


class TESSERA_OT_build_single_object(bpy.types.Operator):
    """Build .nc2 cluster cache for a specific mesh object"""
    bl_idname = "tessera.build_single_object"
    bl_label = "Build Object Cache"
    bl_description = "Generate .nc2 cluster cache for the specified object"
    bl_options = {'REGISTER', 'UNDO'}

    object_name: bpy.props.StringProperty(
        name="Object Name",
        description="Name of the mesh object to build cache for",
        default="",
    )

    def execute(self, context):
        ob = bpy.data.objects.get(self.object_name)
        if not ob and context.active_object:
            ob = context.active_object
        if not ob or ob.type != 'MESH':
            self.report({'WARNING'}, f"Object '{self.object_name}' is not a valid mesh")
            return {'CANCELLED'}

        blend_dir = bpy.path.abspath("//")
        if not blend_dir or not os.path.isdir(blend_dir):
            blend_dir = os.getcwd()

        nc2_path = os.path.join(blend_dir, f"{ob.name}.nc2")
        ob["tessera_nc2"] = nc2_path
        ob["tessera_nc2_path"] = nc2_path
        ob["tessera_enabled"] = 1
        if getattr(ob, "tessera", None):
            ob.tessera.enabled = True

        orig_active = context.view_layer.objects.active
        try:
            context.view_layer.objects.active = ob
            bpy.ops.object.tessera_build(filepath=nc2_path)
        finally:
            context.view_layer.objects.active = orig_active

        self.report({'INFO'}, f"Built cache for '{ob.name}' -> {nc2_path}")
        return {'FINISHED'}


def get_target_nc2_path(ob):
    """Resolve target .nc2 output path for an object."""
    existing = ob.get("tessera_nc2") or ob.get("tessera_nc2_path")
    if existing:
        return existing
    blend_dir = bpy.path.abspath("//")
    if not blend_dir or not os.path.isdir(blend_dir):
        blend_dir = os.getcwd()
    return os.path.join(blend_dir, f"{ob.name}.nc2")


class TESSERA_OT_build_collection(bpy.types.Operator):
    """Build .nc2 cluster caches for all meshes in a collection"""
    bl_idname = "tessera.build_collection"
    bl_label = "Build Collection Caches"
    bl_description = "Generate missing or updated .nc2 cluster caches for all meshes in this collection"
    bl_options = {'REGISTER', 'UNDO'}

    collection_name: bpy.props.StringProperty(
        name="Collection Name",
        description="Name of collection to build caches for",
        default="",
    )

    def execute(self, context):
        col = bpy.data.collections.get(self.collection_name)
        if not col and context.scene:
            if not self.collection_name or self.collection_name == context.scene.collection.name:
                col = context.scene.collection
        if not col:
            self.report({'WARNING'}, f"Collection '{self.collection_name}' not found")
            return {'CANCELLED'}

        meshes = [ob for ob in col.all_objects if ob.type == 'MESH']
        if not meshes:
            self.report({'INFO'}, f"No mesh objects found in collection '{col.name}'")
            return {'FINISHED'}

        built_count = 0
        orig_active = context.view_layer.objects.active
        try:
            for ob in col.all_objects:
                if ob.type != 'MESH':
                    continue

                try:
                    # Ensure object is linked and visible in view layer
                    context.view_layer.objects.active = ob
                    ob.select_set(True)

                    nc2_path = get_target_nc2_path(ob)
                    ob["tessera_nc2"] = nc2_path
                    ob["tessera_nc2_path"] = nc2_path
                    ob["tessera_enabled"] = 1
                    if getattr(ob, "tessera", None):
                        ob.tessera.enabled = True

                    # Use context override for Blender 4.x / 5.x
                    with context.temp_override(active_object=ob, selected_objects=[ob], object=ob):
                        if bpy.ops.object.tessera_build.poll():
                            bpy.ops.object.tessera_build(filepath=nc2_path)
                            built_count += 1
                        else:
                            # Fallback direct call if poll fails
                            print(f"[Tessera] Warning: poll failed for {ob.name}, skipping.")
                except Exception as e:
                    print(f"[Tessera] Warning: failed to build {ob.name}: {e}")
        finally:
            try:
                context.view_layer.objects.active = orig_active
            except Exception:
                pass

        self.report({'INFO'}, f"Built .nc2 caches for {built_count} meshes in '{col.name}'")
        return {'FINISHED'}


def _tag_all_3d_viewports(context):
    """Tag every 3D viewport across all open windows for a redraw."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class TESSERA_OT_collection_set_enabled(bpy.types.Operator):
    """Enable or disable Tessera for all mesh objects inside a collection"""
    bl_idname = "tessera.collection_set_enabled"
    bl_label = "Collection Enable/Disable Tessera"
    bl_description = ("Batch-enable or batch-disable Tessera virtual geometry "
                      "for every mesh object inside this collection, with "
                      "immediate viewport synchronization")
    bl_options = {'REGISTER', 'UNDO'}

    collection_name: bpy.props.StringProperty(
        name="Collection Name",
        description="Name of collection to enable/disable Tessera on",
        default="",
    )
    enabled: bpy.props.BoolProperty(
        name="Enable Tessera",
        description="Enable Tessera for all member meshes when true, disable when false",
        default=True,
    )
    recursive: bpy.props.BoolProperty(
        name="Recursive",
        description="Also affect meshes inside child collections",
        default=True,
    )

    def execute(self, context):
        col = bpy.data.collections.get(self.collection_name)
        if not col and context.scene:
            if not self.collection_name or self.collection_name == context.scene.collection.name:
                col = context.scene.collection
        if not col:
            self.report({'WARNING'}, f"Collection '{self.collection_name}' not found")
            return {'CANCELLED'}

        target_objects = col.all_objects if self.recursive else col.objects

        affected = 0
        for ob in target_objects:
            if ob.type != 'MESH' or ob.data is None:
                continue
            try:
                if hasattr(ob, "tessera"):
                    ob.tessera.enabled = self.enabled
                ob["tessera_enabled"] = int(self.enabled)
                affected += 1
            except Exception:
                # ScatterFlow / procedural assets may hold objects mid-evaluation
                # whose PropertyGroup is not writable; the raw IDProperty fallback
                # is always safe and is what the C++ draw code reads.
                try:
                    ob["tessera_enabled"] = int(self.enabled)
                    affected += 1
                except Exception:
                    continue

        _tag_all_3d_viewports(context)
        state_str = "enabled" if self.enabled else "disabled"
        self.report({'INFO'}, f"Tessera {state_str} for {affected} meshes in '{col.name}'")
        return {'FINISHED'}


class TESSERA_OT_toggle_collection(bpy.types.Operator):
    """Toggle expansion disclosure for collection in Tessera tree"""
    bl_idname = "tessera.toggle_collection"
    bl_label = "Toggle Collection"
    bl_description = "Expand or collapse collection tree"
    bl_options = {'INTERNAL'}

    collection_name: bpy.props.StringProperty(
        name="Collection Name",
        description="Name of collection to toggle",
        default="",
    )

    def execute(self, context):
        col = bpy.data.collections.get(self.collection_name)
        if not col and context.scene:
            if self.collection_name == context.scene.collection.name:
                col = context.scene.collection
        if col:
            current = bool(col.get("tessera_tree_expanded", True))
            col["tessera_tree_expanded"] = not current
            if context.window_manager:
                for window in context.window_manager.windows:
                    if window.screen:
                        for area in window.screen.areas:
                            if area.type == 'VIEW_3D':
                                area.tag_redraw()
        return {'FINISHED'}


class TESSERA_OT_convert_to_proxy(bpy.types.Operator):
    """Convert base mesh to lightweight proxy while retaining .nc2 virtual geometry"""
    bl_idname = "tessera.convert_to_proxy"
    bl_label = "Convert Base Mesh to Lightweight Proxy (Keep .nc2)"
    bl_description = "Replaces the heavy Blender base mesh with an 8-vertex bounding box proxy in memory, eliminating depsgraph CPU evaluation while Tessera renders the full virtual geometry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.ui import get_existing_nc2_path
        from tessera.render import _tag_tessera_redraw

        targets = [ob for ob in context.selected_objects if ob.type == 'MESH']
        if not targets and context.active_object and context.active_object.type == 'MESH':
            targets = [context.active_object]

        count = 0
        for ob in targets:
            nc2 = get_existing_nc2_path(ob)
            if not nc2 or not os.path.exists(nc2):
                continue
            if ob.get("tessera_is_proxy", 0):
                continue

            orig_mesh = ob.data
            ob["tessera_orig_mesh_name"] = orig_mesh.name
            ob["tessera_nc2"] = nc2
            ob["tessera_nc2_path"] = nc2
            ob["tessera_enabled"] = 1
            if getattr(ob, "tessera", None):
                ob.tessera.enabled = True

            bbox = [list(corner) for corner in ob.bound_box]
            faces = [
                (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)
            ]

            proxy_mesh = bpy.data.meshes.new(f"{orig_mesh.name}_Proxy")
            proxy_mesh.from_pydata(bbox, [], faces)
            proxy_mesh.update()

            for mat in orig_mesh.materials:
                proxy_mesh.materials.append(mat)

            ob.data = proxy_mesh
            ob["tessera_is_proxy"] = 1
            count += 1
            _tag_tessera_redraw(ob, context)

        self.report({'INFO'}, f"Converted {count} meshes to lightweight proxies (kept .nc2)")
        return {'FINISHED'}


class TESSERA_OT_restore_mesh(bpy.types.Operator):
    """Restore original heavy mesh from proxy representation"""
    bl_idname = "tessera.restore_mesh"
    bl_label = "Restore Original Mesh"
    bl_description = "Restores the original full-resolution Blender base mesh from the proxy representation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from tessera.render import _tag_tessera_redraw

        targets = [ob for ob in context.selected_objects if ob.type == 'MESH']
        if not targets and context.active_object and context.active_object.type == 'MESH':
            targets = [context.active_object]

        count = 0
        for ob in targets:
            if not ob.get("tessera_is_proxy", 0):
                continue
            orig_name = ob.get("tessera_orig_mesh_name", "")
            orig_mesh = bpy.data.meshes.get(orig_name)
            if orig_mesh:
                proxy_mesh = ob.data
                ob.data = orig_mesh
                ob["tessera_is_proxy"] = 0
                if proxy_mesh and proxy_mesh.users == 0:
                    bpy.data.meshes.remove(proxy_mesh)
                count += 1
                _tag_tessera_redraw(ob, context)

        self.report({'INFO'}, f"Restored {count} original meshes")
        return {'FINISHED'}


classes = (
    TESSERA_OT_batch_enable_all,
    TESSERA_OT_batch_enable_selected,
    TESSERA_OT_batch_disable_selected,
    TESSERA_OT_batch_build_all,
    TESSERA_OT_build_single_object,
    TESSERA_OT_build_collection,
    TESSERA_OT_collection_set_enabled,
    TESSERA_OT_toggle_collection,
    TESSERA_OT_convert_to_proxy,
    TESSERA_OT_restore_mesh,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
