import bpy
import os
from mathutils import Vector
import math
import numpy as np

# ========== CONFIG ==========
obj_folder = "F://TREES//NEW//simplified" # "F:\deciduous\deciduous_simplified" # "F:\conifers\.OBJ CONIFERS\conifers_simplified"
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
template_texture = "textures/coniferous.jpg" # deciduous.jpg" # coniferous.jpg"
template_texture = os.path.join(base_folder, template_texture)
bark_texture = "textures/bark_coniferous.jpg" # deciduous.jpg" # coniferous.jpg"
bark_texture = os.path.join(base_folder, bark_texture)
height_ratio = 0.3  # bottom X% is considered trunk
output_folder = "F://TREES/NEW/TREES_OBJ" # conifers"
os.makedirs(output_folder, exist_ok=True)

# def clear_scene():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)

def clear_scene():
    # Switch to Object mode if not already
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Deselect all
    for obj in bpy.data.objects:
        obj.select_set(False)

    # Select and delete all mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Also remove unused data blocks (optional but helpful)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def import_obj(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        bpy.ops.wm.obj_import(filepath=filepath)
        mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
        if not mesh_objects:
            print(f"No mesh objects found in {filepath}")
            return None
        
        # Return the first mesh object
        imported_obj = mesh_objects[0]
        
        # Validate the imported object
        if not imported_obj.data.vertices:
            print(f"Imported object has no vertices: {filepath}")
            return None
            
        return imported_obj
        
    except Exception as e:
        print(f"Failed to import {filepath}: {e}")
        return None

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
    if not is_valid_mesh_object(obj, "align_base_to_ground obj"):
        return
    
    # Make object active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def split_crown_trunk(obj, max_trunk_height_ratio=0.3):
    if not is_valid_mesh_object(obj, "split_crown_trunk obj"):
        return []
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    align_base_to_ground(obj)

    verts = obj.data.vertices
    z_coords = [v.co.z for v in verts]
    
    if not z_coords:
        print(f"No valid z coordinates found for {obj.name}")
        return []
        
    z_min, z_max = min(z_coords), max(z_coords)
    trunk_height_threshold = z_min + (z_max - z_min) * max_trunk_height_ratio

    trunk_verts = {v.index for v in verts if v.co.z <= trunk_height_threshold}
    crown_verts = {v.index for v in verts if v.co.z > trunk_height_threshold}

    if not crown_verts:
        print(f"No crown vertices found for {obj.name}")
        return []

    for v in verts:
        v.select = (v.index in crown_verts)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

    return bpy.context.selected_objects  # likely [trunk, crown]

def assign_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def ensure_object_context(obj):
    """Ensure proper context for object operations"""
    if not obj or obj.type != 'MESH':
        return False
    
    # Clear all selections
    bpy.ops.object.select_all(action='DESELECT')
    
    # Set active object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Ensure we're in object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    return True

def assign_vertex_colors_from_texture(obj, texture_path):
    if not is_valid_mesh_object(obj, "assign_vertex_colors obj"):
        return
    
    # Load image
    try:
        img = bpy.data.images.load(texture_path)
        img_pixels = np.array(img.pixels[:]).reshape((img.size[1], img.size[0], 4))  # shape: (H, W, RGBA)
    except Exception as e:
        print(f"Failed to load texture {texture_path}: {e}")
        return

    # Ensure UV map exists
    if not obj.data.uv_layers:
        # Store current mode
        current_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
        
        # Clear selection and set active object
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Ensure we're in object mode first
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Enter edit mode and select all
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Validate context before UV operations
        if (bpy.context.object and 
            bpy.context.object.type == 'MESH' and 
            bpy.context.mode == 'EDIT_MESH' and
            len(bpy.context.object.data.polygons) > 0):
            try:
                bpy.ops.uv.smart_project(angle_limit=1.22173, island_margin=0.02)
            except RuntimeError as e:
                print(f"Smart UV projection failed: {e}")
                try:
                    # Fallback to unwrap
                    bpy.ops.uv.unwrap()
                except RuntimeError:
                    print("Unwrap also failed, creating basic UV layer")
                    # Create basic UV coordinates manually
                    bpy.ops.object.mode_set(mode='OBJECT')
                    if not obj.data.uv_layers:
                        obj.data.uv_layers.new(name="UVMap")
        else:
            print(f"Invalid context for UV projection: mode={bpy.context.mode}, polygons={len(obj.data.polygons) if obj.data.polygons else 0}")
        
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    # Ensure vertex color layer exists
    if not obj or not hasattr(obj, 'data') or not obj.data:
        print(f"Object became invalid during processing: {obj}")
        return
        
    if "Col" not in obj.data.vertex_colors:
        obj.data.vertex_colors.new(name="Col")
    color_layer = obj.data.vertex_colors["Col"]

    # Validate UV layers exist
    if not obj.data.uv_layers or not obj.data.uv_layers.active:
        print(f"No valid UV layers found for {obj.name}")
        return
        
    uv_layer = obj.data.uv_layers.active.data

    # Validate polygons exist
    if not obj.data.polygons:
        print(f"No polygons found for {obj.name}")
        return

    # Loop through all polygons and assign vertex colors
    for poly in obj.data.polygons:
        for loop_idx in poly.loop_indices:
            if loop_idx >= len(uv_layer):
                continue  # Skip invalid indices
            uv = uv_layer[loop_idx].uv
            x = min(int(uv.x * img.size[0]), img.size[0] - 1)
            y = min(int(uv.y * img.size[1]), img.size[1] - 1)
            rgba = img_pixels[y, x]
            
            if loop_idx < len(color_layer.data):
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
    if not parts:
        print("No parts provided for crown/trunk labeling")
        return None, None
    
    # Get min Z for each part
    min_z_values = []
    for part in parts:
        if not part or part.type != 'MESH':
            continue
        
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
            return None, None

    # Sort by min Z (lower = trunk)
    sorted_parts = sorted(min_z_values, key=lambda x: x[1])
    trunk, crown = sorted_parts[0][0], sorted_parts[1][0]

    # Rename them
    trunk.name = "Trunk"
    crown.name = "Crown"

    print(f"Assigned: {trunk.name} (lower), {crown.name} (upper)")
    return trunk, crown

def is_valid_mesh_object(obj, check_name="object"):
    """Comprehensive validation for mesh objects"""
    if not obj:
        print(f"❌ {check_name} is None")
        return False
    
    if not hasattr(obj, 'type'):
        print(f"❌ {check_name} has no type attribute")
        return False
        
    if obj.type != 'MESH':
        print(f"❌ {check_name} is not a mesh (type: {obj.type})")
        return False
    
    if not hasattr(obj, 'data') or not obj.data:
        print(f"❌ {check_name} has no mesh data")
        return False
    
    if not hasattr(obj.data, 'vertices') or not obj.data.vertices:
        print(f"❌ {check_name} has no vertices")
        return False
    
    if len(obj.data.vertices) == 0:
        print(f"❌ {check_name} has zero vertices")
        return False
    
    return True

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
    
    try:
        clear_scene()
        obj_path = os.path.join(obj_folder, filename)
        tree = import_obj(obj_path)

        if not is_valid_mesh_object(tree, f"imported tree {filename}"):
            continue

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        # crown_mat = create_material_with_texture(template_texture, "FoliageMat")
        # trunk_mat = create_material_with_texture(bark_texture, "BarkMat")

        parts = split_crown_trunk(tree, height_ratio)
        
        if not parts:
            print(f"Failed to split crown/trunk for {filename}, skipping")
            continue
            
        trunk, crown = label_crown_and_trunk(parts)
        
        if is_valid_mesh_object(crown, f"crown of {filename}") and len(crown.data.vertices) > 5:
            assign_vertex_colors_from_texture(crown, template_texture)
        else:
            print(f"Crown invalid or too few vertices for {filename}")

        if is_valid_mesh_object(trunk, f"trunk of {filename}") and len(trunk.data.vertices) > 5:
            assign_vertex_colors_from_texture(trunk, bark_texture)
        else:
            print(f"Trunk invalid or too few vertices for {filename}")

        # Optional: save or render
        bpy.ops.wm.obj_export(filepath=out_path, export_colors=True)
        print(f"Successfully processed and saved {filename}")

        counter += 1
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

    # break

print("All trees processed.")

# blender --python 1_trees_colors.py