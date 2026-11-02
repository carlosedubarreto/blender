import bpy

for ob_name in ['Gravel_13', 'Ground_005', 'Ground_013', 'Ground_009', 'Ground_001']:
    ob = bpy.data.objects.get(ob_name)
    if ob:
        print(f"{ob.name}: loc={ob.location}, t_en={ob.get('tessera_enabled')}, nc2={ob.get('tessera_nc2')}")

