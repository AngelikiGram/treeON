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

# ========== CONFIG ==========
# pointcloud, dsm FALSE -> TREE
# dsm TRUE -> DSM
# pointcloud TRUE, dsm FALSE -> POINTCLOUD
dir = "C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//outputs/trees-colored"
os.makedirs(dir, exist_ok=True)

pointcloud = True # False
dsm = False # True
model_name = "test3_norm01_colorsrgb"  # "model_twoClasses" or "test3"
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
tree_folder = f"models//{model_name}" # model_twoClasses" 
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = f"models//{model_name}/pointclouds-landmarks"  
    output_folder = f"models/COLORED/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
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

# import bpy
# import pandas as pd


import bpy
import bmesh
import numpy as np
from plyfile import PlyData, PlyElement
from mathutils import Vector, Euler
import math
import os

# Load metadata
csv_path = os.path.join(base_folder, "trees-data.csv")
df = pd.read_csv(csv_path)
# Make ID the index for quick lookup
df["ID"] = df["ID"].astype(str).str.zfill(3)  # zero-pad to match '001' format
category_map = dict(zip(df["ID"], df["Category"]))

# ========== UTILS ==========
def import_mesh(path, matched_texture, matched_texture_bark):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        # bpy.ops.import_scene.obj(filepath=path)
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
        # # import_point_cloud_as_spheres(path, matched_texture, matched_texture_bark)
        # import_point_cloud_as_splats(path, matched_texture, matched_texture_bark)

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
    source_img = io.imread(source_img_file).astype(float) / 255.0
    target_img = io.imread(target_img_file).astype(float) / 255.0
    matched = np.empty_like(source_img)
    for c in range(3):
        matched[:, :, c] = exposure.match_histograms(source_img[:, :, c], target_img[:, :, c])
    io.imsave(output_file, (matched * 255).astype(np.uint8))

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def add_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (5, 5, 5)

from PIL import Image

def get_average_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    np_img = np.asarray(img) / 255.0
    avg = np.mean(np_img.reshape(-1, 3), axis=0)
    return tuple((avg * 255).astype(np.uint8))

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
    bm.free()
    bpy.data.objects.remove(obj, do_unlink=True)

    if not verts:
        print(f"[ERROR] No vertices found in: {ply_path}")
        return

    # Apply rotation
    rot_matrix = Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
    verts = [(rot_matrix @ v.to_4d()).xyz for v in verts]

    # Bounds for trunk/crown split
    xs, ys, zs = zip(*verts)
    z_min, z_max = min(zs), max(zs)
    max_radius = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    max_trunk_height = z_min + (z_max - z_min) * 0.35

    # Load texture colors
    crown_rgb = get_average_rgb(crown_texture_path)
    bark_rgb = get_average_rgb(bark_texture_path)

    colored_points = []
    for v in verts:
        r_xy = math.sqrt(v.x**2 + v.y**2)
        is_trunk = (v.z < max_trunk_height) and (r_xy < 0.2 * max_radius)
        r, g, b = bark_rgb if is_trunk else crown_rgb
        colored_points.append((v.x, v.y, v.z, r, g, b))

    # Save PLY
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
    matched_texture = os.path.join(temp_folder, f"crown.jpg")
    matched_texture_bark = os.path.join(temp_folder, f"{tree_id}-bark.jpg")
    render_output = os.path.join(output_folder, f"{tree_id}.png")

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

    try:
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
        colored_output = os.path.join(output_folder, f"tree_{tree_id}.ply")
        export_colored_pointcloud_from_mesh_coords(tree_path, colored_output, obj, matched_texture, matched_texture_bark)
        print(f"[✓] Processed: {tree_id}")
    except Exception as e:
        print(f"[ERROR] Failed to process {tree_id}: {e}")
        continue

# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py








# ------------------------------------------------------
from plyfile import PlyData, PlyElement
import numpy as np
import argparse


def convert_to_gaussian_ready_ply(input_ply, output_ply):
    plydata = PlyData.read(input_ply)
    old_verts_raw = plydata["vertex"]
    old_verts = old_verts_raw.data  # THIS FIXES THE CRASH

    field_names = old_verts.dtype.names
    num_points = len(old_verts)

    # Get xyz
    xyz = np.stack([old_verts["x"], old_verts["y"], old_verts["z"]], axis=1)


    # Extract RGB from last 3 values of each vertex (e.g., nx, ny, nz, r, g, b)
    # We assume r, g, b are the last 3 float fields
    rgb_fields = field_names[-3:]
    colors = np.stack([old_verts[name] for name in rgb_fields], axis=1).astype(np.float32)

    # Normalize if needed (if values are 0–255)
    if colors.max() > 1.0:
        colors /= 255.0

    # SH DC coefficients
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0

    # Default scale, rotation, opacity
    scales = np.full((num_points, 3), -3.0, dtype=np.float32)  # log-scale
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (num_points, 1)).astype(np.float32)
    opacities = np.zeros((num_points,), dtype=np.float32)
    rgb_uint8 = (colors * 255).astype(np.uint8)

    # Final PLY structure
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
    ]

    vertex_data = np.empty(num_points, dtype=dtype)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["red"] = rgb_uint8[:, 0]
    vertex_data["green"] = rgb_uint8[:, 1]
    vertex_data["blue"] = rgb_uint8[:, 2]
    vertex_data["f_dc_0"] = f_dc[:, 0]
    vertex_data["f_dc_1"] = f_dc[:, 1]
    vertex_data["f_dc_2"] = f_dc[:, 2]
    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]
    vertex_data["rot_0"] = rotations[:, 0]
    vertex_data["rot_1"] = rotations[:, 1]
    vertex_data["rot_2"] = rotations[:, 2]
    vertex_data["rot_3"] = rotations[:, 3]
    vertex_data["opacity"] = opacities

    print('vertex_data', vertex_data["red"], vertex_data["green"], vertex_data["blue"])

    # Save to .ply
    ply_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([ply_element], text=True).write(output_ply)
    print(f"✅ Saved Gaussian-ready PLY to: {output_ply}")

# convert_to_gaussian_splatting_format("models\\test3_norm01\\pointclouds-landmarks\\tree_1.ply", "models\\test3_norm01\\tree_1_splat.ply")
# convert_to_gaussian_splatting_format("C:\\Users\\mmddd\\Downloads\\tree_1.ply", "C:\\Users\\mmddd\\Downloads\\Tree_splat.ply")
# import trimesh
# def convert_ply_to_obj(input_ply, output_obj):
#     mesh = trimesh.load(input_ply)
#     mesh.export(output_obj)
#     print(f"[✓] Converted to: {output_obj}")
# convert_ply_to_obj("C:\\Users\\mmddd\\Downloads\\tree_1.ply", "C:\\Users\\mmddd\\Downloads\\tree_1.obj")

input_folder = output_folder
out_folder = f"models/SPLATS/{model_name}"
os.makedirs(out_folder, exist_ok=True)
for file in os.listdir(input_folder):
    if file.endswith(".ply"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(out_folder, file)
        convert_to_gaussian_ready_ply(input_path, output_path)

# python convert_to_splat.py      


