import bpy

for w in bpy.context.window_manager.windows:
    for a in w.screen.areas:
        if a.type == 'VIEW_3D':
            sp = a.spaces.active
            sh = sp.shading
            print(f"Shading type: {sh.type}, light: {sh.light}, color_type: {sh.color_type}")
            print(f"Shadows: {sh.show_shadows}, cavity: {sh.show_cavity}, outline: {sh.show_object_outline}")
            print(f"Dof: {sh.show_dof}, specular: {sh.show_specular_highlight}")

