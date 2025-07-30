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

tree_main_folder = "C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//TREE_MODELS"
for folder in os.listdir(tree_main_folder):
    tree_folder = os.path.join(tree_main_folder, folder)
    

model_name = "mixed_lessSh" # "_DENSE" #_DENSE-POINTY _mixed-POINTY (this we want with mixed for MS) # "mixed"  # TOCHANGE
splats = False # True
pointcloud = True # False
dsm = True # False # True

# ========== CONFIG ==========
# pointcloud, dsm FALSE -> TREE
# dsm TRUE -> DSM
# pointcloud TRUE, dsm FALSE -> POINTCLOUD
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
tree_folder = f"TREE_MODELS//{model_name}"
if splats:
    tree_folder = f"TREE_MODELS//SPLATS//{model_name}"
output_folder = f"outputs/trees-meshes/{model_name}"
if pointcloud: 
    tree_folder = f"TREE_MODELS//{model_name}/pointclouds-landmarks"  
    output_folder = f"outputs/{model_name}"
    if splats:
        tree_folder = f"TREE_MODELS//SPLATS//{model_name}//"
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
os.makedirs(output_folder, exist_ok=True)
temp_folder = os.path.join(base_folder, temp_folder)
# template_texture = os.path.join(texture_folder, "coniferous.jpg")  # or deciduous.jpg
render_resolution = (512, 512) # (512, 512) # (220, 220)
background_color = (1, 1, 1, 1)  # White RGB

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

def import_mesh(path, matched_texture, matched_texture_bark):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        # bpy.ops.import_scene.obj(filepath=path)
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        if splats: 
            import_point_cloud_as_splats(path)
            # Find the node group
            for ng in bpy.data.node_groups:
                if "GaussianSplatting" in ng.name:
                    print("Found node group:", ng.name)
                    # Access a node inside it
                    for node in ng.nodes:
                        print("  Contains node:", node.name)
                    # Example: access specific node by partial name
                    target_node = next((n for n in ng.nodes if "Boolean" in n.name), None)
                    if target_node:
                        print("  Found Boolean node:", target_node.name)
                    bpy.data.node_groups["GaussianSplatting"].nodes["Boolean"].boolean = False
                    break
        else: 
            import_point_cloud_as_spheres(path, matched_texture, matched_texture_bark)

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
    if splats: 
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
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
    bpy.ops.object.import_gaussian_splatting(filepath=ply_path)
    bpy.context.object.rotation_euler[0] += math.radians(90)

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

    base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//'
    # foliage_mat = create_textured_material(os.path.join(base_folder, "textures/coniferous.jpg"), "FoliageMat")
    foliage_mat = create_textured_material(matched_texture, "FoliageMat")
    # bark_mat = create_textured_material(os.path.join(base_folder, "textures/bark_coniferous.jpg"), "BarkMat")
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
    mesh = bpy.data.meshes.new("MergedPointCloudMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("MergedPointCloudMesh", mesh)
    obj.data.materials.append(foliage_mat)
    obj.data.materials.append(bark_mat)
    bpy.context.collection.objects.link(obj)

    # # Create output collection
    # sphere_collection = bpy.data.collections.new("PointSpheres")
    # bpy.context.scene.collection.children.link(sphere_collection)

    # # Place spheres with logic
    # for i, coord in enumerate(verts):
    #     r = math.sqrt(coord.x ** 2 + coord.y ** 2)
    #     is_trunk = (coord.z < max_trunk_height) and (r < 0.2 * max_radius)
    #     base = bark_sphere if is_trunk else foliage_sphere
    #     inst = bpy.data.objects.new(f"pt_{i}", base.data)
    #     inst.location = coord
    #     sphere_collection.objects.link(inst)

    # # Join all
    # for obj in sphere_collection.objects:
    #     obj.select_set(True)
    # bpy.context.view_layer.objects.active = sphere_collection.objects[0]
    # bpy.ops.object.join()
    # bpy.context.object.name = "MergedPointCloudMesh"
    # # bpy.context.object.rotation_euler[0] += math.radians(90)

def import_point_cloud_as_spheres1(ply_path, radius=0.15): # 0.05
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
    # bpy.context.object.rotation_euler[0] += math.radians(90)
    bpy.context.object.name = "MergedPointCloudMesh"

# ========== MAIN ==========
tree_main_folder = "C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//TREE_MODELS"
for folder in os.listdir(tree_main_folder):
    if folder != model_name:
        print(f"Skipping folder {folder}, not matching model name {model_name}")
        continue
    tree_folder = os.path.join(tree_main_folder, folder)
    tree_folder = os.path.join(tree_folder, 'pointclouds-landmarks')
    model_name = folder
    output_folder = f"outputs/{model_name}"
    output_folder = os.path.join(base_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)

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
        ortho_path = os.path.join(ortho_folder, f"ortho_{tree_id}.png")
        matched_texture = os.path.join(temp_folder, f"crown.jpg")
        matched_texture_bark = os.path.join(temp_folder, f"{tree_id}-bark.jpg")
        render_output = os.path.join(output_folder, f"{tree_id}.png")
        if os.path.exists(render_output): #  or '50' in render_output:
            print(f"Skipping {tree_id}, already rendered")
            # continue

        # target_ids = ['69'] # ['10', '11', '12', '13', '17', '18', '19', '21', '22', '24', '26', '28', '29', '30', '31', '32', '33', '34', '35', '36', '38', '55', '57', '61', '63', '67', '68', '69', '72']
        # if not any(tid in render_output for tid in target_ids):
        #     print(f"Skipping {tree_id}")
        #     continue

        # if '13' not in render_output:
        #     print(f"Skipping {tree_id}, already rendered")
        #     continue

        # if not os.path.exists(ortho_path):
        #     print(f"Skipping {tree_id}, ortho not found")
        #     continue

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

            setup_camera(obj)
            add_light()
            setup_render(render_output)
            bpy.ops.render.render(write_still=True)
            print("Rendered", tree_id)
        except Exception as e:
            print(f"Error processing {tree_id}: {e}")
            continue

# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings.py









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
