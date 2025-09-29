import sys
import site

# Manually add the user site-packages directory to sys.path
site_path = site.getusersitepackages()
if site_path not in sys.path:
    sys.path.append(site_path)
import os
import numpy as np
from skimage import io, exposure
import math

model_name = "mixed_all" # _noCl"  # TOCHANGE
pointcloud = True # False
dsm = False # True

# "TREE_MODELS/COLORED/" contains colored point clouds
# Saved in 'outputs' folder

# ========== CONFIG ==========
base_folder = 'C://Users//mmddd//Documents//network-tree-gen//landmarks_austria//'
tree_folder = f"TREE_MODELS//{model_name}" 
tree_folder = os.path.join(base_folder, tree_folder)
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = f"TREE_MODELS//{model_name}/pointclouds-landmarks"  
    tree_folder = os.path.join(base_folder, tree_folder)
    output_folder = f"TREE_MODELS/COLORED/{model_name}"
    output_folder = os.path.join(base_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)
if dsm: 
    tree_folder = "DATA_LANDMARKS/DSM_OBJ"  
    output_folder = f"outputs/dsm/{model_name}"
ortho_folder = "DATA_LANDMARKS/ORTHOPHOTOS"
texture_folder = "textures"
temp_folder = "temp"
tree_folder = os.path.join(base_folder, tree_folder)
ortho_folder = os.path.join(base_folder, ortho_folder)
texture_folder = os.path.join(base_folder, texture_folder)
output_folder = os.path.join(base_folder, output_folder)
temp_folder = os.path.join(base_folder, temp_folder)
render_resolution = (220, 220) # (512, 512) # (250, 250)
background_color = (1, 1, 1, 1)  # White RGBA

import bpy
import bmesh
import numpy as np
from plyfile import PlyData, PlyElement
from mathutils import Vector, Euler
import math
import os

import pandas as pd

# Load metadata
csv_path = os.path.join(base_folder, "trees-data.csv")
df = pd.read_csv(csv_path)
# Make ID the index for quick lookup
df["ID"] = df["ID"].astype(str).str.zfill(3)  # zero-pad to match '001' format
category_map = dict(zip(df["ID"], df["Category"]))

# ========== UTILS ==========
def import_mesh(path, matched_texture, matched_texture_bark, output_path=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        # bpy.ops.import_scene.obj(filepath=path)
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        if output_path:
            # Use the new colored point cloud function
            import_point_cloud_as_colored_points(path, matched_texture, matched_texture_bark, output_path)
            return None  # No object to return since we export directly
        else:
            bpy.ops.wm.ply_import(filepath=path)

from mathutils import Vector
def setup_camera(target_obj, distance=75.0): # 50.0):
    # Compute object's bounding box center in world coordinates
    bbox_corners = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    bbox_center = sum(bbox_corners, Vector()) / 8.0

    # Place camera in front of the object (on +X axis), looking toward the center
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_location = bbox_center + Vector((distance, 0, 0))  # in front of the object along X
    cam_obj.location = cam_location

    # Point the camera to look at the object's center
    direction = bbox_center - cam_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set this camera as the active one
    bpy.context.scene.camera = cam_obj
def match_histogram_texture(source_img_file, target_img_file, output_file):
    try:
        source_img = io.imread(source_img_file).astype(float) / 255.0
        target_img = io.imread(target_img_file).astype(float) / 255.0
        matched = np.empty_like(source_img)
        for c in range(3):
            matched[:, :, c] = exposure.match_histograms(source_img[:, :, c], target_img[:, :, c])
        io.imsave(output_file, (matched * 255).astype(np.uint8))
    except Exception as e:
        print(f"[WARNING] Failed to match histogram: {e}")
        # If histogram matching fails, just copy the source file
        try:
            source_img = io.imread(source_img_file)
            io.imsave(output_file, source_img)
            print(f"[INFO] Used original texture instead: {source_img_file}")
        except Exception as e2:
            print(f"[ERROR] Failed to copy source texture: {e2}")
            raise e2

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def add_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (5, 5, 5)

from PIL import Image

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

def sample_texture_color(image_path, uv_coord):
    """Sample color from texture at UV coordinate"""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        x = int(uv_coord[0] * width) % width
        y = int((1.0 - uv_coord[1]) * height) % height  # Flip Y for UV coordinates
        return img.getpixel((x, y))
    except Exception as e:
        print(f"[WARNING] Failed to sample texture {image_path}: {e}")
        # If the file cannot be opened, check if it exists and is a valid image
        if not os.path.isfile(image_path):
            print(f"[ERROR] Texture file does not exist: {image_path}")
        else:
            print(f"[ERROR] Texture file is not a valid image: {image_path}")
        # Return a default color (green for crown, brown for bark)
        if "bark" in image_path.lower():
            return (101, 67, 33)  # Brown
        else:
            return (34, 139, 34)  # Forest green

def get_uv_coordinate(vertex, bounds):
    """Generate UV coordinate based on vertex position"""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    # Map X,Y position to UV (0-1 range)
    u = (vertex.x - x_min) / (x_max - x_min) if x_max != x_min else 0.5
    v = (vertex.y - y_min) / (y_max - y_min) if y_max != y_min else 0.5
    return (u, v)

def export_colored_pointcloud_from_mesh_coords(ply_path, output_path, obj, crown_texture_path, bark_texture_path, radius=0.14):
    import math
    from mathutils import Euler
    import numpy as np
    import bmesh
    from plyfile import PlyData, PlyElement

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    verts = [v.co.copy() for v in bm.verts]
    bpy.data.objects.remove(obj, do_unlink=True)
    # Only free bm after all mesh operations are complete
    bm.free()

    if not verts:
        print(f"[ERROR] No vertices found in: {ply_path}")
        return

    # Apply rotation
    rot_matrix = Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
    verts = [(rot_matrix @ v.to_4d()).xyz for v in verts]

    # Bounds for trunk/crown split and UV mapping
    xs, ys, zs = zip(*verts)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    max_radius = 0.5 * max(x_max - x_min, y_max - y_min)
    max_trunk_height = z_min + (z_max - z_min) * 0.35

    colored_points = []
    for v in verts:
        r_xy = math.sqrt(v.x**2 + v.y**2)
        is_trunk = (v.z < max_trunk_height) and (r_xy < 0.2 * max_radius)
        
        # Generate UV coordinate and sample texture
        uv_coord = get_uv_coordinate(v, bounds)
        texture_path = bark_texture_path if is_trunk else crown_texture_path
        r, g, b = sample_texture_color(texture_path, uv_coord)
        colored_points.append((v.x, v.y, v.z, r, g, b))

    # Save PLY
    vertex_array = np.array(
        colored_points,
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    PlyData([PlyElement.describe(vertex_array, 'vertex')], text=True).write(output_path)
    print(f"[✓] Exported colored point cloud to: {output_path}")

def import_point_cloud_as_colored_points(ply_path, matched_texture, matched_texture_bark, output_path, radius=0.14):
    """Import PLY point cloud and export it with correct colors from textures"""
    import bpy, bmesh, math
    import numpy as np
    from mathutils import Vector
    import mathutils
    from plyfile import PlyData, PlyElement

    # Import PLY
    bpy.ops.wm.ply_import(filepath=ply_path)
    point_cloud = bpy.context.selected_objects[0]
    point_cloud.name = "PointCloud"

    # Extract and rotate points
    mesh = point_cloud.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    verts = [v.co.copy() for v in bm.verts]
    bpy.data.objects.remove(point_cloud, do_unlink=True)
    # Only free bm after all mesh operations are complete
    bm.free()

    if len(verts) == 0:
        print(f"[WARNING] No vertices found in: {ply_path}")
        return

    # Apply rotation
    rot_matrix = mathutils.Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
    verts = [(rot_matrix @ v.to_4d()).xyz for v in verts]

    # Bounds for trunk/crown split and UV mapping
    xs, ys, zs = zip(*verts)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    max_radius = 0.5 * max(x_max - x_min, y_max - y_min)
    max_trunk_height = z_min + (z_max - z_min) * 0.35

    colored_points = []
    for v in verts:
        r_xy = math.sqrt(v.x**2 + v.y**2)
        is_trunk = (v.z < max_trunk_height) and (r_xy < 0.2 * max_radius)
        
        # Generate UV coordinate and sample texture
        uv_coord = get_uv_coordinate(v, bounds)
        texture_path = matched_texture_bark if is_trunk else matched_texture
        r, g, b = sample_texture_color(texture_path, uv_coord)
        colored_points.append((v.x, v.y, v.z, r, g, b))

    # Save colored PLY
    vertex_array = np.array(
        colored_points,
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    PlyData([PlyElement.describe(vertex_array, 'vertex')], text=True).write(output_path)
    print(f"[✓] Exported colored point cloud to: {output_path}")


# ========== MAIN ==========
for fname in os.listdir(tree_folder):
    print(fname)
    if fname.endswith("pointclouds") or fname.endswith("mtl") or fname.endswith("export-ply"):
        continue

    tree_id = os.path.splitext(fname)[0]
    tree_id = os.path.splitext(fname)[0].split('_')[1] 
    tree_path = os.path.join(tree_folder, fname)
    ortho_path = os.path.join(ortho_folder, f"ortho_{tree_id}.png")
    matched_texture = os.path.join(temp_folder, f"{tree_id}-crown.jpg")
    matched_texture_bark = os.path.join(temp_folder, f"{tree_id}-bark.jpg")
    render_output = os.path.join(output_folder, f"{tree_id}.png")

    tree_id = os.path.splitext(fname)[0].split('_')[1]  # gets '001' from 'tree_001_mesh.obj'
    category = category_map.get(tree_id, "Deciduous").strip().lower()
    if category == "coniferous":
        template_texture = os.path.join(texture_folder, "coniferous.jpg")
    else:
        template_texture = os.path.join(texture_folder, "deciduous.jpg")

    target_ids = ['5', '10', '11', '12', '13', '15', '17', '18', '22', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34', '35', '36', '38', '55', '57', '61', '72', '69', '67', '68']
    if not any(tid in render_output for tid in target_ids):
        print(f"Skipping {tree_id}")
        # continue

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

    # Skip if output already exists
    colored_output = os.path.join(output_folder, f"tree_{tree_id}.ply")
    if os.path.exists(colored_output):
        print(f"[SKIP] Output already exists: {tree_id}")
        continue

    try:
        # Skip if output already exists
        colored_output = os.path.join(output_folder, f"tree_{tree_id}.ply")
        
        # For PLY files, use direct colored point cloud processing
        if tree_path.lower().endswith('.ply'):
            import_point_cloud_as_colored_points(tree_path, matched_texture, matched_texture_bark, colored_output)
            print(f"[✓] Processed: {tree_id}")
            continue
        
        # For OBJ files, use the original mesh approach
        import_mesh(tree_path, matched_texture, matched_texture_bark)

        obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
        obj.location = (0, 0, 0)
        obj.scale = (1, 1, 1)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        if pointcloud == False:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')

        # setup_camera(obj)
        # add_light()
        export_colored_pointcloud_from_mesh_coords(tree_path, colored_output, obj, matched_texture, matched_texture_bark)
        print(f"[✓] Processed: {tree_id}")
    except Exception as e:
        print(f"[ERROR] Failed to process {tree_id}: {e}")
        continue

# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py






