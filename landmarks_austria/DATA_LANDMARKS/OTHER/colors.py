import bpy
import os
from mathutils import Vector
import math
import numpy as np

# ========== CONFIG ==========
obj_folder = "F:\conifers\.OBJ CONIFERS\conifers_simplified"
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
template_texture = "textures/coniferous.jpg"
template_texture = os.path.join(base_folder, template_texture)
bark_texture = "textures/bark_coniferous.jpg"
bark_texture = os.path.join(base_folder, bark_texture)
height_ratio = 0.3  # bottom X% is considered trunk
output_folder = "F://TREES/conifers"
os.makedirs(output_folder, exist_ok=True)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_obj(filepath):
    bpy.ops.wm.obj_import(filepath=filepath)
    return [obj for obj in bpy.context.selected_objects if obj.type == 'MESH'][0]

def create_material_with_texture(img_path, name):
    img = bpy.data.images.load(img_path)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    return mat

import bmesh
def align_base_to_ground(obj):
    # Make object active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def split_crown_trunk(obj, max_trunk_radius=2.0, max_trunk_height_ratio=0.35, step=0.2):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    align_base_to_ground(obj)

    verts = obj.data.vertices
    z_coords = [v.co.z for v in verts]
    z_min, z_max = min(z_coords), max(z_coords)
    max_trunk_height = z_min + (z_max - z_min) * max_trunk_height_ratio

    trunk_verts = set()
    crown_verts = set()

    z = z_min
    while z < z_max:
        ring = [v for v in verts if z <= v.co.z < z + step]
        if not ring:
            z += step
            continue

        avg_radius = np.mean([math.sqrt(v.co.x ** 2 + v.co.y ** 2) for v in ring])
        is_trunk = (avg_radius < max_trunk_radius) and (z < max_trunk_height)

        if is_trunk:
            trunk_verts.update(v.index for v in ring)
        else:
            crown_verts.update(v.index for v in ring)

        z += step

    # Mark crown vertices for separation
    for v in verts:
        v.select = (v.index in crown_verts)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

    return bpy.context.selected_objects  # likely [trunk, crown]

def assign_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def assign_vertex_colors_from_texture(obj, texture_path):
    # Ensure we're in object mode and the object is active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Load image
    img = bpy.data.images.load(texture_path)
    img_pixels = np.array(img.pixels[:]).reshape((img.size[1], img.size[0], 4))  # shape: (H, W, RGBA)

    # Ensure UV map exists
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')

    # Ensure vertex color layer exists
    if "Col" not in obj.data.vertex_colors:
        obj.data.vertex_colors.new(name="Col")
    color_layer = obj.data.vertex_colors["Col"]

    uv_layer = obj.data.uv_layers.active.data

    # Loop through all polygons and assign vertex colors
    for poly in obj.data.polygons:
        for loop_idx in poly.loop_indices:
            uv = uv_layer[loop_idx].uv
            x = min(int(uv.x * img.size[0]), img.size[0] - 1)
            y = min(int(uv.y * img.size[1]), img.size[1] - 1)
            rgba = img_pixels[y, x]
            color_layer.data[loop_idx].color = rgba

    # Assign a material that uses vertex colors
    mat = bpy.data.materials.new(name="VertexColorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    vcol_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    vcol_node.layer_name = "Col"
    mat.node_tree.links.new(vcol_node.outputs['Color'], bsdf.inputs['Base Color'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    print("Vertex colors assigned from texture.")

def label_crown_and_trunk(parts):
    # Get min Z for each part
    min_z_values = []
    for part in parts:
        verts = part.data.vertices
        if len(verts) == 0:
            continue  # Skip empty objects
        min_z = min((v.co.z for v in verts), default=float('inf'))
        min_z_values.append((part, min_z))

    if len(min_z_values) < 2:
        # If only one part has vertices, label it as Crown
        if len(min_z_values) == 1:
            sole_part = min_z_values[0][0]
            sole_part.name = "Crown"
            print(f"Only one valid part found: {sole_part.name}")
            return None, sole_part
        else:
            print("⚠️ No valid parts with vertices found.")
            return None, None

    # Sort by min Z (lower = trunk)
    sorted_parts = sorted(min_z_values, key=lambda x: x[1])
    trunk, crown = sorted_parts[0][0], sorted_parts[1][0]

    # Rename them
    trunk.name = "Trunk"
    crown.name = "Crown"

    print(f"Assigned: {trunk.name} (lower), {crown.name} (upper)")
    return trunk, crown

# ========== MAIN ==========
counter = 0

import random

# Count already existing .obj files in the output folder
existing_outputs = [f for f in os.listdir(output_folder) if f.endswith(".obj")]
remaining_slots = max(0, 1000 - len(existing_outputs))
print(f"Already processed: {len(existing_outputs)}, Remaining slots: {remaining_slots}")

# Get all .obj files from the input folder
all_obj_files = [f for f in os.listdir(obj_folder) if f.endswith(".obj")]
random.shuffle(all_obj_files)

# Filter out the ones already processed
all_obj_files = [f for f in all_obj_files if not os.path.exists(os.path.join(output_folder, f))]

# Take only the required number
selected_files = all_obj_files[:remaining_slots]


for filename in selected_files:
# os.listdir(obj_folder):
    if not filename.endswith(".obj"):
        continue

    out_path = os.path.join(output_folder, filename)
    if os.path.exists(out_path):
        print(f"Skipping {filename} (already exists)")
        continue

    print(f"Processing {filename}")
    clear_scene()
    obj_path = os.path.join(obj_folder, filename)
    tree = import_obj(obj_path)

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    # crown_mat = create_material_with_texture(template_texture, "FoliageMat")
    # trunk_mat = create_material_with_texture(bark_texture, "BarkMat")

    parts = split_crown_trunk(tree, height_ratio)
    trunk, crown = label_crown_and_trunk(parts)
    if crown and len(crown.data.vertices) > 0:
        assign_vertex_colors_from_texture(crown, template_texture)

    if trunk and len(trunk.data.vertices) > 0:
        assign_vertex_colors_from_texture(trunk, bark_texture)

    # Optional: save or render
    bpy.ops.wm.obj_export(filepath=out_path, export_colors=True)

    counter += 1

    # break

print("All trees processed.")