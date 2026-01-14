import bpy
import os
import numpy as np
from skimage import io, exposure
import math

# ========== CONFIG ==========
# pointcloud, dsm FALSE -> TREE
# dsm TRUE -> DSM
# pointcloud TRUE, dsm FALSE -> POINTCLOUD
pointcloud = True # False
dsm = False # True
model_name = "test3"  # "model_twoClasses" or "test3"
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
tree_folder = f"models//{model_name}" # model_twoClasses" 
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = f"models//{model_name}/pointclouds"  
    output_folder = f"outputs/trees-pointclouds/{model_name}"
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
render_resolution = (512, 512) # (250, 250)
background_color = (1, 1, 1, 1)  # White RGBA

import pandas as pd

# Load metadata
csv_path = os.path.join(base_folder, "trees-data.csv")
df = pd.read_csv(csv_path)
# Make ID the index for quick lookup
df["ID"] = df["ID"].astype(str).str.zfill(3)  # zero-pad to match '001' format
category_map = dict(zip(df["ID"], df["Category"]))

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

def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        # bpy.ops.import_scene.obj(filepath=path)
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        import_point_cloud_as_spheres(path)
        # bpy.ops.wm.ply_import(filepath=path)

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

def setup_camera(target_obj, distance=50.0): # 50.0):
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

def setup_render(path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    scene.render.resolution_x, scene.render.resolution_y = render_resolution
    scene.render.film_transparent = False  # use True for transparent background
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = background_color

    # Set render engine to 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
    # Enable transparent background
    bpy.context.scene.render.film_transparent = True  # For Cycles
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Ensure alpha channel is saved
    bpy.context.scene.render.image_settings.file_format = 'PNG'

def add_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (5, 5, 5)

import bmesh
def import_point_cloud_as_spheres(ply_path, radius=0.15):
    bpy.ops.wm.ply_import(filepath=ply_path)
    point_cloud_obj = bpy.context.selected_objects[0]
    point_cloud_obj.name = "PointCloud"

    # Extract vertex positions
    mesh = point_cloud_obj.data
    bm_in = bmesh.new()
    bm_in.from_mesh(mesh)
    verts = [v.co.copy() for v in bm_in.verts]
    bm_in.free()
    
    bpy.data.objects.remove(point_cloud_obj, do_unlink=True)

    if len(verts) == 0:
        print(f"[WARNING] No vertices found in point cloud: {ply_path}")
        return

    # Create a sphere template
    template_bm = bmesh.new()
    bmesh.ops.create_uvsphere(template_bm, u_segments=6, v_segments=4, radius=radius)

    # Create final bmesh
    bm_out = bmesh.new()
    for coord in verts:
        temp = template_bm.copy()
        for v in temp.verts:
            v.co += coord

        temp_mesh = bpy.data.meshes.new("temp")
        temp.to_mesh(temp_mesh)
        bm_out.from_mesh(temp_mesh)
        bpy.data.meshes.remove(temp_mesh)
        temp.free()

    # Final mesh object
    mesh_data = bpy.data.meshes.new("PointCloudSpheres")
    bm_out.to_mesh(mesh_data)
    bm_out.free()

    obj = bpy.data.objects.new("PointCloudSpheres", mesh_data)
    bpy.context.collection.objects.link(obj)

    # Assign black material
    mat = bpy.data.materials.get("BlackMaterial")
    if mat is None:
        mat = bpy.data.materials.new("BlackMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
            bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.append(mat)

    print(f"[INFO] Created {len(verts)} spheres as a single mesh")
 
def import_point_cloud_as_spheres1(ply_path, radius=0.05):
    # Import the PLY file
    bpy.ops.wm.ply_import(filepath=ply_path)

    # Create a black material
    black_mat = bpy.data.materials.get("BlackMaterial")
    if black_mat is None:
        black_mat = bpy.data.materials.new(name="BlackMaterial")
        black_mat.use_nodes = True
        nodes = black_mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)  # RGBA black
            bsdf.inputs["Roughness"].default_value = 1.0

    point_cloud = bpy.context.selected_objects[0]
    point_cloud.name = "PointCloud"

    # Extract vertex positions
    mesh = point_cloud.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    verts = [v.co.copy() for v in bm.verts]
    bm.free()

    if len(verts) == 0:
        print(f"[WARNING] No vertices found in point cloud: {ply_path}")
        bpy.data.objects.remove(point_cloud, do_unlink=True)
        return

    # Remove original point cloud object
    bpy.data.objects.remove(point_cloud, do_unlink=True)

    # Create sphere mesh for instancing
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
    sphere = bpy.context.object
    sphere.name = "PointSphere"
    sphere.hide_render = True
    sphere.hide_viewport = True

    # Create a collection for the spheres
    sphere_collection = bpy.data.collections.new("PointSpheres")
    bpy.context.scene.collection.children.link(sphere_collection)

    # Add sphere copies at each point location
    for i, coord in enumerate(verts):
        inst = bpy.data.objects.new(f"pt_{i}", sphere.data)
        inst.location = coord
        sphere_collection.objects.link(inst)

        # Assign the black material
        if len(inst.data.materials) == 0:
            inst.data.materials.append(black_mat)
        else:
            inst.data.materials[0] = black_mat

    # Remove the template sphere
    bpy.data.objects.remove(sphere, do_unlink=True)

    # Only proceed if spheres were added
    if len(sphere_collection.objects) == 0:
        print(f"[WARNING] No spheres created for point cloud: {ply_path}")
        return

    # Select and join all spheres
    for obj in sphere_collection.objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = sphere_collection.objects[0]
    bpy.ops.object.join()

    # Rotate 90 degrees around X
    # bpy.context.object.rotation_euler[0] += math.radians(90) ## 
    bpy.context.object.name = "MergedPointCloudMesh"

# ========== MAIN ==========
for fname in os.listdir(tree_folder):
    print(fname)
    if fname.endswith("pointclouds") or fname.endswith("mtl") or fname.endswith("export-ply"):
        continue
    # if not fname.endswith(".obj"):
        # continue

    tree_id = os.path.splitext(fname)[0]
    tree_id = os.path.splitext(fname)[0].split('_')[1] + os.path.splitext(fname)[0].split('_')[2] ## 
    tree_path = os.path.join(tree_folder, fname)
    ortho_path = os.path.join(ortho_folder, f"ortho_{tree_id}.png")
    matched_texture = os.path.join(temp_folder, f"{tree_id}-crown.jpg")
    render_output = os.path.join(output_folder, f"{tree_id}.png")
    if os.path.exists(render_output) or '50' in render_output:
        print(f"Skipping {tree_id}, already rendered")
        continue

    # if not os.path.exists(ortho_path):
    #     print(f"Skipping {tree_id}, ortho not found")
    #     continue

    tree_id = os.path.splitext(fname)[0].split('_')[1]  # gets '001' from 'tree_001_mesh.obj'
    category = category_map.get(tree_id, "Deciduous").strip().lower()
    if category == "coniferous":
        template_texture = os.path.join(texture_folder, "coniferous.jpg")
    else:
        template_texture = os.path.join(texture_folder, "deciduous.jpg")

#    match_histogram_texture(template_texture, ortho_path, matched_texture) ##

    clear_scene()
    try:
        import_mesh(tree_path)

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

    #  apply_texture(obj, matched_texture) ##
        setup_camera(obj)
        add_light()
        setup_render(render_output)
        bpy.ops.render.render(write_still=True)
        print("Rendered", tree_id)
    except Exception as e:
        print(f"Error processing {tree_id}: {e}")
        continue

# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings.py