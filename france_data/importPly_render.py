import bpy
import os

RADIUS = 0.0075 # 15

script_dir1 = 'C:/Users/mmddd/Documents/P2/COMPARISONS_MODELS/FRANCE/output_4.ply' # query_points COMPARISONS_MODELS

# RADIUS = 0.255
# script_dir1 = 'C:/Users/mmddd/Desktop/P@/France/tree_1.ply'


# Set output render dimensions
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1920
bpy.context.scene.render.film_transparent = True

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.node_groups:
        bpy.data.node_groups.remove(block)

def setup_camera(obj):
    for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
        bpy.data.objects.remove(cam, do_unlink=True)
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not bbox:
        print('No vertices found for camera placement.')
        return
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)
    center = obj.location
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    max_size = max(size_x, size_y, size_z)
    cam_distance = max_size * 4.5
    cam_obj.location = (center.x, center.y - cam_distance, (min_z + max_z) / 2)
    cam_obj.rotation_euler = (1.5708, 0, 0)
    bpy.context.scene.camera = cam_obj

def add_sun_light(obj):
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not bbox:
        print('No vertices found for sun placement.')
        return
    light_data = bpy.data.lights.new(name="TreeKeyLight", type='SUN')
    light_obj = bpy.data.objects.new(name="TreeKeyLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    light_obj.location = (center_x, center_y - (max_y - min_y) * 2, center_z + (max_z - min_z) * 2)
    import mathutils
    tree_center = mathutils.Vector((center_x, center_y, center_z))
    light_direction = tree_center - light_obj.location
    light_obj.rotation_euler = light_direction.to_track_quat('Z', 'Y').to_euler()
    light_obj.rotation_euler.x = 1.5708

def import_ply_as_spheres(ply_path, sphere_radius=RADIUS, use_colors=False):
    points = []
    colors = []
    with open(ply_path, 'r') as f:
        header = True
        has_colors = False
        for line in f:
            if header:
                if 'property uchar red' in line or 'property uchar green' in line or 'property uchar blue' in line:
                    has_colors = True
                if line.strip() == 'end_header':
                    header = False
                continue
            vals = line.strip().split()
            if len(vals) >= 3:
                x, y, z = map(float, vals[:3])
                points.append((x, y, z))
                if has_colors and len(vals) >= 6:
                    r, g, b = int(vals[3]), int(vals[4]), int(vals[5])
                    colors.append((r/255.0, g/255.0, b/255.0, 1.0))
                else:
                    colors.append((0.15, 0.15, 0.15, 1.0))  # Default gray
    mesh = bpy.data.meshes.new("SpheresMesh")
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new("SpheresObj", mesh)
    bpy.context.collection.objects.link(obj)
    
    # Create vertex colors if colors are available
    if has_colors and colors:
        color_attr = mesh.color_attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
        for i, color in enumerate(colors):
            if i < len(color_attr.data):
                color_attr.data[i].color = color
    
    geo_mod = obj.modifiers.new("SpheresGeo", 'NODES')
    node_group = bpy.data.node_groups.new("SpheresGeoTree", 'GeometryNodeTree')
    geo_mod.node_group = node_group
    nodes = node_group.nodes
    links = node_group.links
    nodes.clear()
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    node_group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Subdivisions"].default_value = 1
    ico.inputs["Radius"].default_value = sphere_radius
    shade_smooth = nodes.new("GeometryNodeSetShadeSmooth")
    shade_smooth.inputs["Shade Smooth"].default_value = True
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    set_material = nodes.new("GeometryNodeSetMaterial")
    
    # Create material that uses vertex colors if available
    sphere_mat = bpy.data.materials.new("SphereColorMat")
    sphere_mat.use_nodes = True
    mat_nodes = sphere_mat.node_tree.nodes
    mat_links = sphere_mat.node_tree.links
    
    # Get or create nodes
    bsdf = mat_nodes.get('Principled BSDF')
    # if has_colors and colors:
    #     # Add Attribute node to read vertex colors
    #     attr_node = mat_nodes.new('ShaderNodeAttribute')
    #     attr_node.attribute_name = "Color"
    #     attr_node.attribute_type = 'GEOMETRY'
    #     # Connect vertex color to base color
    #     mat_links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    # else:
    #     # Use 0.15 gray color
    #     bsdf.inputs['Base Color'].default_value = (0.15, 0.15, 0.15, 1.0)

    bsdf.inputs['Base Color'].default_value = (0.15, 0.15, 0.15, 1.0)
    
    set_material.inputs["Material"].default_value = sphere_mat
    links.new(input_node.outputs["Geometry"], instance.inputs["Points"])
    links.new(ico.outputs["Mesh"], shade_smooth.inputs["Geometry"])
    links.new(shade_smooth.outputs["Geometry"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])
    
    # Set origin to geometry center, then reset location and rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    
    return obj

def main():
    clear_scene()
    obj = import_ply_as_spheres(script_dir1, sphere_radius=RADIUS)
    
    setup_camera(obj)
    add_sun_light(obj)

    obj.location.x -= 0.5
    # Optionally render and save image
    # render_and_save('output_image.png')

if __name__ == "__main__":
    main()
