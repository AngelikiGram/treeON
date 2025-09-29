import sys
import site
site_path = site.getusersitepackages()
if site_path not in sys.path:
    sys.path.append(site_path)
import os
import numpy as np
from skimage import io, exposure

import bpy
import os
import numpy as np
import math

model_name = "test"  
splats = False # True

# ========== CONFIG ==========
# pointcloud, dsm FALSE -> TREE
# dsm TRUE -> DSM
# pointcloud TRUE, dsm FALSE -> POINTCLOUD
pointcloud = True # False
dsm = False # True
base_folder = 'C://Users//mmddd//Documents//P2//network-tree-gen//landmarks_austria//'
tree_folder = f"models//{model_name}"
if splats:
    tree_folder = f"models//SPLATS//{model_name}"
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = fr"C:\Users\mmddd\Documents\P2\supersample_pointclouds\OUTPUTS\colored-{model_name}"      
    output_folder = f"outputs/trees-pointclouds/{model_name}"
    if splats:
        tree_folder = f"models//SPLATS//{model_name}//"
        os.makedirs(os.path.join(base_folder, "outputs/trees-pointclouds/SPLATS"), exist_ok=True)
        output_folder = f"outputs/trees-pointclouds/SPLATS/{model_name}"
if dsm: 
    tree_folder = "DSM_OBJ"  
    output_folder = f"outputs/dsm/{model_name}"
ortho_folder = "ORTHOPHOTOS"
texture_folder = "textures"
temp_folder = "temp"
tree_folder = os.path.join(base_folder, tree_folder)
ortho_folder = os.path.join(base_folder, ortho_folder)
texture_folder = os.path.join(base_folder, texture_folder)
output_folder = os.path.join(base_folder, output_folder)
temp_folder = os.path.join(base_folder, temp_folder)
# template_texture = os.path.join(texture_folder, "coniferous.jpg")  # or deciduous.jpg
render_resolution = (220, 220) # (512, 512) # (250, 250)
background_color = (1, 1, 1, 1)  # White RGBA


ROTATE_PC_X90 = False  # rotate point clouds around X by +90° if needed
import mathutils
from mathutils import Vector, Euler

import os
import csv
# Load metadata
csv_path = os.path.join(base_folder, "trees-data.csv")
category_map = {}
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tree_id = row["ID"].zfill(3)  # zero-pad to match '001' format
        category = row["Category"]
        category_map[tree_id] = category

output_folder = os.path.join(base_folder, "temp/scenes")
os.makedirs(output_folder, exist_ok=True)
DSM_FOLDER = os.path.join(output_folder, "DSM_TIF")
ORTHO_FOLDER = os.path.join(output_folder, "ORTHOPHOTOS")

out_folder_scenes = os.path.join(base_folder, "outputs", "SCENES")
os.makedirs(out_folder_scenes, exist_ok=True)

# import pandas as pd

# # Load metadata
# csv_path = os.path.join(base_folder, "trees-data.csv")
# df = pd.read_csv(csv_path)
# # Make ID the index for quick lookup
# df["ID"] = df["ID"].astype(str).str.zfill(3)  # zero-pad to match '001' format
# category_map = dict(zip(df["ID"], df["Category"]))

# ========== UTILS ==========

def match_histogram_texture(source_img_file, target_img_file, output_file):
    source_img = io.imread(source_img_file).astype(float) / 255.0
    target_img = io.imread(target_img_file).astype(float) / 255.0
    matched = np.empty_like(source_img)
    for c in range(3):
        matched[:, :, c] = exposure.match_histograms(source_img[:, :, c], target_img[:, :, c])
    io.imsave(output_file, (matched * 255).astype(np.uint8))

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def first_mesh():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return meshes[0] if meshes else None

def import_mesh(path, matched_texture, matched_texture_bark):
    ext = os.path.splitext(path)[1].lower()
    obj = None
    if ext == ".obj":
        # bpy.ops.import_scene.obj(filepath=path)
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        if splats: 
            obj = import_point_cloud_as_splats(path)
        else: 
            obj = import_mesh_obj(path, texture_img=matched_texture, dsm=False) # import_point_cloud_as_spheres(path, matched_texture, matched_texture_bark)

def apply_texture(obj, img_path, mat_name="CrownColors"):
    img = bpy.data.images.load(img_path)

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img

    links.new(tex.outputs["Color"], diffuse.inputs["Color"])
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # UV unwrap
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

from mathutils import Vector
import bpy

def get_z_at_xy(obj, x, y):
    """
    Finds the Z coordinate of the vertex on the given object `obj` 
    that is closest in (x, y) to the provided coordinates.
    
    Args:
        obj (bpy.types.Object): The mesh object to search.
        x (float): X coordinate.
        y (float): Y coordinate.
        
    Returns:
        float: The Z coordinate of the closest (x, y) match.
    """
    if obj.type != 'MESH':
        raise ValueError("Object must be a mesh")

    min_dist = float("inf")
    closest_z = None

    world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    for v in world_verts:
        dist = (v.x - x) ** 2 + (v.y - y) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_z = v.z

    return closest_z

def setup_camera(target_obj, dsm_obj, distance=75.0, offset_above_dsm=4.0):

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.data.lens = 35

    # Compute the object's bounding box center in world coordinates
    bbox_corners = [Vector(corner) for corner in target_obj.bound_box] # target_obj.matrix_world @ 
    for i, corner in enumerate(bbox_corners):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=corner) # 0.2
        sphere = bpy.context.object
        sphere.name = f"BBoxCorner_{i}"
    bbox_center = sum(bbox_corners, Vector()) / 8.0
    # Get the maximum Z value of DSM (world coords)
    dsm_z_vals = [(dsm_obj.matrix_world @ Vector(corner)).z for corner in dsm_obj.bound_box]
    dsm_max_z = max(dsm_z_vals)
    # Camera Z: max(tree Z center, DSM max Z + offset)
    cam_z = get_z_at_xy(dsm_obj, bbox_center.x + distance, bbox_center.y) + offset_above_dsm
    print(f"Camera Z:", bbox_center.x + distance, bbox_center.y)

    # Place the camera in front along X, at chosen Z height
    cam_location = Vector((bbox_center.x + distance, bbox_center.y, cam_z))
    cam_obj.location = cam_location

    # Make the camera look at the center of the object
    direction = bbox_center - cam_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    for i, corner in enumerate(bbox_corners):
        name = f"BBoxCorner_{i}"
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

def setup_render(path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    scene.render.resolution_x, scene.render.resolution_y = render_resolution
    scene.render.film_transparent = False  # use True for transparent background
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    # bg = world.node_tree.nodes["Background"]
    # bg.inputs[0].default_value = background_color

    # Set render engine to 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
    if splats: 
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # Enable transparent background
    # bpy.context.scene.render.film_transparent = True  # For Cycles
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Ensure alpha channel is saved
    bpy.context.scene.render.image_settings.file_format = 'PNG'

def add_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (5, 5, 5)

    light.rotation_euler = (1.2, 0.1, 0.4)  # Rotate to point downwards

def setup_sky():
    # Use nodes for world
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create new nodes
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    bg_node = nodes.new(type="ShaderNodeBackground")
    sky_node = nodes.new(type="ShaderNodeTexSky")

    # Set positions for clarity
    output_node.location = (400, 0)
    bg_node.location = (200, 0)
    sky_node.location = (0, 0)

    # Connect nodes: Sky -> Background -> World Output
    links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # Optional: set sun position or type
    sky_node.sky_type = 'HOSEK_WILKIE'  # Better looking sky model
    sky_node.sun_elevation = 0.6      # Between 0 (horizon) and 1 (zenith)
    sky_node.sun_rotation = 1.0       # Around vertical axis

import bmesh
def create_sphere_template(radius=0.14):
    import bpy
    import bmesh

    mesh = bpy.data.meshes.new("TempSphere")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    return mesh
def import_point_cloud_as_splats(ply_path):
    # bpy.ops.object.import_gaussian_splatting(filepath=ply_path)
    # obj = bpy.context.object
    # obj.rotation_euler[0] += math.radians(90)

    bpy.ops.sna.import_ply_as_splats_8458e(filepath=ply_path)
    bpy.context.object.sna_kiri3dgs_active_object_update_mode = 'Enable Camera Updates'
    bpy.data.materials["KIRI_3DGS_Render_Material"].node_tree.nodes["Group.005"].inputs[3].default_value = 1
    # obj = bpy.context.object
    for object_scene in bpy.data.objects:
        if 'tree' in object_scene.name.lower():
            obj = object_scene
            break
    # obj.rotation_euler[0] += math.radians(90)

    return obj


def create_textured_material(img_path, name):
        if name in bpy.data.materials:
            return bpy.data.materials[name]
        img = bpy.data.images.load(img_path)
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        for n in nodes: nodes.remove(n)

        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = img
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.new("ShaderNodeOutputMaterial")

        # Links
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        return mat

def load_points_with_colors(path):
    ext = os.path.splitext(path)[1].lower()
    verts, cols = [], []
    if ext == ".ply":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if not f.readline().startswith("ply"):
                raise ValueError("Not a PLY file.")
            format_ascii = False
            nverts = None
            prop = []
            in_vert = False
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF in PLY header.")
                s = line.strip()
                if s.startswith("format"):
                    format_ascii = "ascii" in s
                    if not format_ascii:
                        raise ValueError("Only ASCII PLY is supported.")
                elif s.startswith("element"):
                    parts = s.split()
                    in_vert = (len(parts) == 3 and parts[1] == "vertex")
                    if in_vert:
                        nverts = int(parts[2])
                elif s.startswith("property") and in_vert:
                    prop.append(s.split()[-1])
                elif s == "end_header":
                    break
            ix, iy, iz = prop.index("x"), prop.index("y"), prop.index("z")
            ir, ig, ib = prop.index("red"), prop.index("green"), prop.index("blue")
            for _ in range(nverts):
                parts = f.readline().split()
                if len(parts) < len(prop): continue
                x, y, z = float(parts[ix]), float(parts[iy]), float(parts[iz])
                r, g, b = int(parts[ir]), int(parts[ig]), int(parts[ib])
                verts.append(Vector((x, y, z)))
                cols.append((r/255.0, g/255.0, b/255.0, 1.0))
    elif ext == ".obj":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.strip().split()
                    if len(p) >= 7:
                        x, y, z = map(float, p[1:4])
                        r, g, b = map(float, p[4:7])
                        verts.append(Vector((x, y, z)))
                        cols.append((r, g, b, 1.0))
    else:
        raise ValueError(f"Unsupported point format: {ext}")
    if ROTATE_PC_X90:
        R = Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
        verts = [(R @ v.to_4d()).xyz for v in verts]
    return verts, cols

def import_point_cloud_as_spheres(ply_path, matched_texture, matched_texture_bark, radius=0.14): # 0.15
    import bpy, bmesh, math
    import numpy as np
    from mathutils import Vector
    import mathutils

    # Import PLY
    bpy.ops.wm.ply_import(filepath=ply_path)
    point_cloud = bpy.context.selected_objects[0]
    point_cloud.name = "PointCloud"

    # Extract and rotate points
    mesh = point_cloud.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    verts = [v.co.copy() for v in bm.verts]
    bm.free()
    bpy.data.objects.remove(point_cloud, do_unlink=True)

    if len(verts) == 0:
        print(f"[WARNING] No vertices found in: {ply_path}")
        return


    rot_matrix = mathutils.Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
    for v in verts:
        v_rot = rot_matrix @ v.to_4d()
        v.x, v.y, v.z = v_rot.xyz

    # Compute bounds
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    z_min, z_max = min(zs), max(zs)
    max_radius = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    max_trunk_height = z_min + (z_max - z_min) * 0.35

    foliage_mat = create_textured_material(matched_texture, "FoliageMat")
    bark_mat = create_textured_material(matched_texture_bark, "BarkMat")

    # Create template spheres
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
    foliage_sphere = bpy.context.object
    foliage_sphere.name = "FoliageSphere"
    foliage_sphere.hide_render = True
    foliage_sphere.hide_viewport = True
    foliage_sphere.data.materials.append(foliage_mat)

    bark_sphere = foliage_sphere.copy()
    bark_sphere.data = foliage_sphere.data.copy()
    bark_sphere.name = "BarkSphere"
    bark_sphere.data.materials.clear()
    bark_sphere.data.materials.append(bark_mat)
    bpy.context.collection.objects.link(bark_sphere)

    sphere_mesh = create_sphere_template(radius)
    bm = bmesh.new()

    for coord in verts:
        r = math.sqrt(coord.x ** 2 + coord.y ** 2)
        is_trunk = (coord.z < max_trunk_height) and (r < 0.2 * max_radius)
        mat_index = 1 if is_trunk else 0

        mat = bark_mat if is_trunk else foliage_mat

        # Duplicate base sphere and move it
        temp_bm = bmesh.new()
        # bmesh.ops.create_uvsphere(temp_bm, u_segments=16, v_segments=8, diameter=radius)
        bmesh.ops.create_uvsphere(temp_bm, u_segments=16, v_segments=8, radius=radius)
        bmesh.ops.translate(temp_bm, verts=temp_bm.verts, vec=coord)

        # Assign material index to all faces
        for f in temp_bm.faces:
            f.material_index = mat_index

        # bm.from_mesh(temp_bm.to_mesh())
        temp_mesh = bpy.data.meshes.new("Temp")
        temp_bm.to_mesh(temp_mesh)
        bm.from_mesh(temp_mesh)
        bpy.data.meshes.remove(temp_mesh)  # optional cleanup
        temp_bm.free()

    # Output final mesh
    mesh = bpy.data.meshes.new("tree")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("tree", mesh)
    obj.data.materials.append(foliage_mat)
    obj.data.materials.append(bark_mat)
    bpy.context.collection.objects.link(obj)

    return obj

#####            
# ========== Load DSM ==========
def load_dsm_as_mesh(dsm_path, mesh_name="DSM_Mesh", mode='DEM_RAW'):
    bpy.ops.importgis.georaster(importMode=mode, filepath=dsm_path)
    dsm_obj = bpy.context.selected_objects[0]
    dsm_obj.name = mesh_name

# ========== Apply Orthophoto ==========
def createTileUVMap(obj_name):
    for object_scene in bpy.data.objects:
        if obj_name in object_scene.name:
            obj = object_scene
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    if not obj.data.uv_layers.active:
        # bpy.ops.uv.reset()

        bpy.ops.uv.unwrap(method='ANGLE_BASED')

        # bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001) # ANGLE_BASED
        
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    min_x = min(v.co.x for v in obj.data.vertices)
    max_x = max(v.co.x for v in obj.data.vertices)
    min_y = min(v.co.y for v in obj.data.vertices)
    max_y = max(v.co.y for v in obj.data.vertices)

    uv_layer = bm.loops.layers.uv.verify()

    # adjust UVs
    for f in bm.faces:
        for l in f.loops:
            luv = l[uv_layer]
            if luv.select:
                # apply the location of the vertex as a UV        
                u = (l.vert.co.x - min_x) / (max_x - min_x)
                v = (l.vert.co.y - min_y) / (max_y - min_y)
                
                luv.uv = (u,v)

    bmesh.update_edit_mesh(me)

    bpy.ops.object.mode_set(mode='OBJECT')

def apply_orthophoto_texture(obj, image_path):
    # Load the image
    img = bpy.data.images.load(image_path)

    # Create a new material
    mat = bpy.data.materials.new(name="Orthophoto_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = img
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Assign the material
    obj.data.materials.append(mat)

    set_specular_to_zero(mat)

def get_lowest_point(dsm_object):
    # object_data = dsm_object.data
    # vertices = object_data.vertices
    # min_z = 100000000

    # lowest_vertex = Vector((0,0,0))

    # for v in vertices:
    #     if v.co.z > min_z:
    #         continue
    #     min_z = v.co.z
    #     lowest_vertex.x = v.co.x
    #     lowest_vertex.y = v.co.y
    #     lowest_vertex.z = v.co.z
    
    # return lowest_vertex

    if obj.type != 'MESH':
        print("Not a mesh object.")
        return

    # Make sure object is active and in Object mode
    bpy.context.view_layer.objects.active = dsm_object
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Apply scale to use transformed geometry
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Move origin to (0,0,0) if needed
    dsm_object.location = (0, 0, 0)

    # Switch to Edit mode to get mesh data
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(dsm_object.data)

    # Get world positions of all vertices
    vertex_world_positions = [obj.matrix_world @ v.co for v in bm.verts]

    bpy.ops.object.mode_set(mode='OBJECT')  # Back to Object mode

    # Find the lowest vertex
    min_z = float('inf')
    lowest_vertex = None
    for pos in vertex_world_positions:
        if pos.z < min_z:
            min_z = pos.z
            lowest_vertex = pos

    # if lowest_vertex:
    #     bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=lowest_vertex)
    #     sphere = bpy.context.object
    #     sphere.name = "LowestVertexSphere"

    return lowest_vertex

def move_tree_to_dsm_surface(tree_obj, dsm_obj):
    # Step 2: Get tree's (x,y) location
    tree_xy = Vector((tree_obj.location.x, tree_obj.location.y))

    # Step 3: Find closest z on DSM mesh
    dsm_z = get_z_at_xy(dsm_obj, tree_xy.x, tree_xy.y)
    print(f"Tree XY: {tree_xy}, DSM Z: {dsm_z}")

    # Step 4: Move tree so origin (lowest point) sits at (x, y, z) on DSM
    tree_obj.location.z = dsm_z

def convert_to_mesh(obj):
    # Get evaluated object with all modifiers and transformations applied
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh().copy()

    # Create a new mesh object
    new_obj = bpy.data.objects.new(obj.name + "_mesh", mesh)
    bpy.context.collection.objects.link(new_obj)

    # Optional: preserve transformation
    new_obj.matrix_world = obj.matrix_world.copy()

    # Free memory
    eval_obj.to_mesh_clear()

    # Remove original object
    bpy.data.objects.remove(obj, do_unlink=True)

    return new_obj


def create_gn_pointcloud(verts, colors, radius, lit=True):
    # make mesh with only vertices + POINT color attribute "Col"
    mesh = bpy.data.meshes.new("PointCloudMesh")
    obj  = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata([tuple(v) for v in verts], [], [])
    mesh.update()

    col_attr = mesh.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
    n = len(mesh.vertices)
    colors = (colors + [(1,1,1,1)])[:n] if len(colors) < n else colors
    for i in range(n):
        col_attr.data[i].color = colors[i]

    # Geometry Nodes: Mesh->Points + Set Material
    mod = obj.modifiers.new(name="GN_Points", type='NODES')
    ng = bpy.data.node_groups.new("GN_PointCloud", 'GeometryNodeTree')
    mod.node_group = ng
    nodes, links = ng.nodes, ng.links
    nodes.clear()

    gin  = nodes.new('NodeGroupInput')
    gout = nodes.new('NodeGroupOutput')
    try:
        ng.interface.new_socket("Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except TypeError:
        pass

    n_m2p  = nodes.new('GeometryNodeMeshToPoints')
    n_setm = nodes.new('GeometryNodeSetMaterial')

    # point size
    if "Radius" in n_m2p.inputs: n_m2p.inputs["Radius"].default_value = radius
    elif "Size" in n_m2p.inputs:  n_m2p.inputs["Size"].default_value   = radius

    # material that reads "Col"
    mat = bpy.data.materials.new("PointColMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        if nd.type != 'OUTPUT_MATERIAL': nt.nodes.remove(nd)
    out = nt.nodes["Material Output"]
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "Col"
    if lit:
        diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
        nt.links.new(attr.outputs["Color"], diff.inputs["Color"])
        nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
    else:
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    n_setm.inputs['Material'].default_value = mat
    links.new(gin.outputs['Geometry'], n_m2p.inputs['Mesh'])
    links.new(n_m2p.outputs['Points'], n_setm.inputs['Geometry'])
    links.new(n_setm.outputs['Geometry'], gout.inputs['Geometry'])
    return obj

def set_specular_to_zero(material):
    if material and material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                if "Specular" in node.inputs:
                    node.inputs["Specular"].default_value = 0.0
                if "Roughness" in node.inputs:
                    node.inputs["Roughness"].default_value = 0.0
                if "IOR" in node.inputs:
                    node.inputs["IOR"].default_value = 1.0
# ========== MAIN ==========
for fname in os.listdir(tree_folder):
    print(fname)
    print('tree_folder', tree_folder)
    if fname.endswith("pointclouds") or fname.endswith("mtl") or fname.endswith("export-ply"):
        continue
    # if not fname.endswith(".obj"):
        # continue

    tree_id = os.path.splitext(fname)[0]
    tree_id = os.path.splitext(fname)[0].split('_')[1] 
    tree_path = os.path.join(tree_folder, fname)

    if not os.path.exists(tree_path):   
        continue

    ortho_path = os.path.join(ortho_folder, f"ortho_{tree_id}.png")
    matched_texture = os.path.join(temp_folder, f"crown.jpg")
    matched_texture_bark = os.path.join(temp_folder, f"{tree_id}-bark.jpg")
    render_output = os.path.join(out_folder_scenes, f"{tree_id}.png")

    tree_id = os.path.splitext(fname)[0].split('_')[1]  # gets '001' from 'tree_001_mesh.obj'
    category = category_map.get(tree_id, "Deciduous").strip().lower()
    if category == "coniferous":
        template_texture = os.path.join(texture_folder, "coniferous.jpg")
    else:
        template_texture = os.path.join(texture_folder, "deciduous.jpg")

    try:
        match_histogram_texture(template_texture, ortho_path, matched_texture)
    except Exception as e:
        print(f"Error matching histogram for {tree_id}: {e}")

    if category == "coniferous":
        template_texture_bark = os.path.join(texture_folder, "bark_coniferous.jpg")
    else:
        template_texture_bark = os.path.join(texture_folder, "bark_deciduous.jpg")
    try:
        # match_histogram_texture(template_texture_bark, ortho_path, matched_texture_bark)
        matched_texture_bark = template_texture_bark
    except Exception as e:
        print(f"Error matching histogram for {tree_id}: {e}")

    clear_scene()
    print('matched_texture', matched_texture)
    # try:
    if 1==1:
        verts, cols = load_points_with_colors(tree_path)
        if not verts:
            print(f"WARNING: No points in {fname}; skipping.")
            continue
        # Compute bounding box size for point cloud
        coords = np.array([v.to_tuple() for v in verts])
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        tree_size = np.linalg.norm(max_xyz - min_xyz)
        point_radius = max(tree_size * 0.0005, 0.01)  # 1% of size, min fallback
        print('point radius:', point_radius)
        obj = create_gn_pointcloud(verts, cols, radius=point_radius, lit=True)

        # Step 1: Put 3D cursor at lowest point of tree, then set origin to it
        lowest_point = get_lowest_point(obj)
        bpy.context.scene.cursor.location = lowest_point
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')
        
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        if pointcloud == False:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')

        tree_id = tree_id.zfill(3)

        load_dsm_as_mesh(os.path.join(DSM_FOLDER, f"tree_{tree_id}_final.tif"), mesh_name=f"DSM_{tree_id}")
        for object_scene in bpy.data.objects:
            if f"DSM" in object_scene.name:
                dsm_obj = object_scene
                break
        createTileUVMap(dsm_obj.name)        
        ortho_path_terrain = os.path.join(ORTHO_FOLDER, f"tree_{tree_id}.png")
        apply_orthophoto_texture(dsm_obj, ortho_path_terrain)
        dsm_obj.select_set(True)
        bpy.context.view_layer.objects.active = dsm_obj
        bpy.ops.object.shade_smooth()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        z_dsm = dsm_obj.location.z
        dsm_obj.location = (0, 0, 0)

        # LARGE DSM
        load_dsm_as_mesh(os.path.join(DSM_FOLDER, f"tree_{tree_id}_large_terrain.tif"), mesh_name=f"Large_{tree_id}", mode='DEM')
        for object_scene in bpy.data.objects:
            if f"Large" in object_scene.name:
                dsm_large_obj = object_scene
                break
        createTileUVMap(dsm_large_obj.name)        
        ortho_path_large_terrain = os.path.join(ORTHO_FOLDER, f"tree_{tree_id}_large.png")
        apply_orthophoto_texture(dsm_large_obj, ortho_path_large_terrain)
        dsm_large_obj.select_set(True)
        bpy.context.view_layer.objects.active = dsm_large_obj
        dsm_large_obj.location = (0, 0, -z_dsm)

        # Bring tree to be on top of DSM at the same xy but change z so that it is above the DSM
        move_tree_to_dsm_surface(obj, dsm_obj)

        # Step 1: Put 3D cursor at lowest point of tree, then set origin to i
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        setup_camera(obj, dsm_obj)
        add_light()
        setup_sky()
        setup_render(render_output)
        bpy.ops.render.render(write_still=True)
        print("Rendered", tree_id)

        # save scene 
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_folder_scenes, f"{tree_id}.blend"))
    # except Exception as e:
    #     print(f"Error processing {tree_id}: {e}")
    #     continue

        # break



# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings_scene.py









# from PIL import Image
# import os

# # Set input and output folders
# input_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//outputs/pictures'
# output_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//outputs/pictures-resized'

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Loop over all image files
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path)

#         # Resize to 250x250
#         img_resized = img.resize((250, 112)) # 250)) # , Image.ANTIALIAS)

#         # Save resized image to output folder
#         output_path = os.path.join(output_folder, filename)
#         img_resized.save(output_path)

# print("All images have been resized to 250x250 pixels.")
