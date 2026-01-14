import bpy
import bmesh
import os
import random
from mathutils import Vector
import math
from scipy.spatial import cKDTree
import numpy as np

# === Settings ===
INPUT_DIR = "C:/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/TREE_MODELS/mixed/pointclouds-landmarks/"
OUTPUT_DIR = "C:/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/TREE_MODELS/_DENSE/pointclouds-landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_NUM_POINTS = 100000  # Total points after densification
K_NEIGHBORS = 150            # Interpolate between k neighbors

def densify_pointcloud(original_points, target_num, k=3):
    """Interpolate between nearby points to generate more points"""
    if len(original_points) >= target_num:
        return original_points[:target_num]

    points = np.array([[v.x, v.y, v.z] for v in original_points])
    tree = cKDTree(points)

    new_points = original_points.copy()
    while len(new_points) < target_num:
        idx = random.randint(0, len(points) - 1)
        _, neighbors_idx = tree.query(points[idx], k=k+1)
        neighbors_idx = neighbors_idx[1:]  # exclude self
        neighbor = points[random.choice(neighbors_idx)]
        alpha = random.random()
        interp = (1 - alpha) * points[idx] + alpha * neighbor
        new_points.append(Vector(interp.tolist()))

    return new_points

def create_point_cloud(name, points):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

# === Clear scene ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === Main loop ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(".ply"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(f"Processing {filename}...")
        try:
            bpy.ops.wm.ply_import(filepath=input_path)
            obj = bpy.context.selected_objects[0]

            obj.rotation_euler[0] += 1.5708

            # Ensure mesh is single-user before applying transform
            obj.data = obj.data.copy()

            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Get original points
            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            original_points = [v.co.copy() for v in bm.verts]
            bm.free()

            # Densify
            dense_points = densify_pointcloud(original_points, TARGET_NUM_POINTS, K_NEIGHBORS)

            # Create new dense point cloud object
            dense_obj = create_point_cloud(obj.name + "_dense", dense_points)

            # Rotate back to original orientation
            dense_obj.rotation_euler[0] -= 1.5708   
            dense_obj.data = dense_obj.data.copy()  # Ensure single-user mesh
            bpy.context.view_layer.objects.active = dense_obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # remove original object
            bpy.data.objects.remove(obj, do_unlink=True)

            # Export
            bpy.context.view_layer.objects.active = dense_obj
            bpy.ops.object.select_all(action='DESELECT')
            dense_obj.select_set(True)
            bpy.ops.wm.ply_export(filepath=output_path)

            print('Exported:', output_path)

            # Cleanup
            # bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.objects.remove(dense_obj, do_unlink=True)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("All files processed.")






import bpy
import bmesh
from mathutils import Vector
import os

# === Settings ===
INPUT_DIR = "C:/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/TREE_MODELS/mixed/pointclouds-landmarks/" # _DENSE
OUTPUT_DIR = "C:/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/TREE_MODELS/_mixed-POINTY/pointclouds-landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOP_FRACTION = 0.05
TAPER_STRENGTH = 0.6

# === Function to make top pointy ===
def make_top_pointy(obj, top_fraction=0.2, taper_strength=2.5):
    # Switch to object mode and make sure it's a mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Compute bounding box height
    zs = [v.co.z for v in bm.verts]
    z_min = min(zs)
    z_max = max(zs)
    height = z_max - z_min
    z_threshold = z_max - top_fraction * height

    # Compute object center in XY (average of all vertices)
    center_xy = Vector((0.0, 0.0))
    for v in bm.verts:
        center_xy += Vector((v.co.x, v.co.y))
    center_xy /= len(bm.verts)

    for v in bm.verts:
        if v.co.z > z_threshold:
            factor = (v.co.z - z_threshold) / (z_max - z_threshold)
            # Pull towards center in XY
            v.co.x -= (v.co.x - center_xy.x) * taper_strength * factor
            v.co.y -= (v.co.y - center_xy.y) * taper_strength * factor
            # Push slightly upwards to sharpen
            v.co.z += taper_strength * factor * 0.1 * height

    # Apply changes
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

# === Clear existing scene ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === Load and process each .ply file ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(".ply"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Import .ply
        print(f"Processing {input_path}...")
        try:
            bpy.ops.wm.ply_import(filepath=input_path)
            obj = bpy.context.selected_objects[0]
            # Rotate 90 degrees around X axis
            obj.rotation_euler[0] += 1.5708  # 90 degrees in radians
            # Apply all transforms
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, isolate_users=True)

            # Apply transformation
            make_top_pointy(obj, TOP_FRACTION, TAPER_STRENGTH)

            # Rotate 90 degrees around X axis
            obj.rotation_euler[0] -= 1.5708  # 90 degrees in radians
            # Apply all transforms
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Export to output folder
            bpy.ops.wm.ply_export(filepath=output_path)

            # Clean up
            bpy.ops.object.delete()
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

print("All files processed and saved.")