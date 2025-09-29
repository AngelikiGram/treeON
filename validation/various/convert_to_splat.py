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

model_name = "test3_norm01_colorsrgb"  # TOCHANGE

# ========== CONFIG ==========
dir = "C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//outputs/trees-colored"
os.makedirs(dir, exist_ok=True)
pointcloud = True # False
dsm = False # True
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
render_resolution = (220, 220) # (512, 512) # (250, 250)
background_color = (1, 1, 1, 1)  # White RGBA

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
    print(f"Saved Gaussian-ready PLY to: {output_ply}")

# convert_to_gaussian_splatting_format("models\\test3_norm01\\pointclouds-landmarks\\tree_1.ply", "models\\test3_norm01\\tree_1_splat.ply")
# convert_to_gaussian_splatting_format("C:\\Users\\mmddd\\Downloads\\tree_1.ply", "C:\\Users\\mmddd\\Downloads\\Tree_splat.ply")
# import trimesh
# def convert_ply_to_obj(input_ply, output_obj):
#     mesh = trimesh.load(input_ply)
#     mesh.export(output_obj)
#     print(f"[✓] Converted to: {output_obj}")
# convert_ply_to_obj("C:\\Users\\mmddd\\Downloads\\tree_1.ply", "C:\\Users\\mmddd\\Downloads\\tree_1.obj")

# convert_to_gaussian_ready_ply("C:\\Users\\mmddd\\Desktop\\untitled.ply", "C:\\Users\\mmddd\\Desktop\\tree_1.ply")

input_folder = output_folder
out_folder = f"models/SPLATS/{model_name}"
out_folder = os.path.join(base_folder, out_folder)
os.makedirs(out_folder, exist_ok=True)
for file in os.listdir(input_folder):
    if file.endswith(".ply"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(out_folder, file)
        convert_to_gaussian_ready_ply(input_path, output_path)

# python convert_to_splat.py      


