import bpy

for ob in bpy.data.objects:
    for m in ob.modifiers:
        if m.type == 'NODES' and hasattr(m, 'properties') and hasattr(m.properties, 'inputs'):
            inputs = m.properties.inputs
            s125 = getattr(inputs, 'Socket_125', None)
            s344 = getattr(inputs, 'Socket_344', None)
            s346 = getattr(inputs, 'Socket_346', None)
            v125 = getattr(s125, 'value', 'N/A') if s125 else 'N/A'
            v344 = getattr(s344, 'value', 'N/A') if s344 else 'N/A'
            v346 = getattr(s346, 'value', 'N/A') if s346 else 'N/A'
            if v125 != 'N/A' or v344 != 'N/A' or v346 != 'N/A':
                print(f"Ob '{ob.name}': Camera Culling={v125}, Dist Culling={v344}, Occlusion Culling={v346}")

