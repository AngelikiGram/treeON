import bpy
import os
import random
import math
import mathutils

import bmesh 

import subprocess


# Set the viewport background and render to white
bpy.context.preferences.themes[0].view_3d.space.gradients.high_gradient = (1.0, 1.0, 1.0)
# bpy.context.preferences.themes[0].view_3d.space.gradients.low_gradient = (1.0, 1.0, 1.0)
bpy.context.scene.display.shading.color_type = 'SINGLE'
bpy.context.scene.display.shading.light = 'FLAT'
bpy.context.scene.display.shading.single_color = (1.0, 1.0, 1.0)

# Override viewport render color to white
# bpy.context.scene.display.shading.material_type = 'SINGLE'
# bpy.context.scene.display.shading.studio_light = 'flat'
bpy.context.scene.display.shading.background_type = 'VIEWPORT'
bpy.context.scene.display.shading.background_color = (1.0, 1.0, 1.0)

print("Viewport and render settings adjusted to white background.")

# Set background color to white
bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
# Set the strength of the background to 1 (optional)
bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = 0.0 # 1.

def set_white_background():
    bpy.context.scene.render.film_transparent = False  # Disable transparency

    # Use World Shader to set the background to white
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    if not world.use_nodes:
        world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes in the World shader
    for node in nodes:
        nodes.remove(node)

    # Add a Background node with white color
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.inputs[0].default_value = (1, 1, 1, 1)  # White color
    background_node.inputs[1].default_value = 0.0  # Strength

    # Add a World Output node
    world_output = nodes.new(type='ShaderNodeOutputWorld')

    # Link the background node to the world output
    links.new(background_node.outputs[0], world_output.inputs[0])

    print("✅ White background applied without affecting scene lighting.")

# Call the function
# set_white_background()

# Render the scene
# bpy.ops.render.render(write_still=True)

import sys
sys.path.append("C:/Users/mmddd/Downloads/dataset-generation-scritps/shadows-gen")

# from shadow_detection import *

# === USER CONFIGURATIONS ===
base_folder = 'C://Users//mmddd//Documents//p2-tree-gen//landmarks_austria//DATA-GEN'
OBJ_FOLDER = "F:/TREES/NEW/TREES_OBJ" # TREES/TREES-temp" # NEW/TREES_OBJ" # TREES-temp" # deciduous" # conifers"  # Path to OBJ tree files
DTM_FOLDER = "ORTHOPHOTOS//dtms"  # Path to DTM (terrain) files
DTM_FOLDER = os.path.join(base_folder, DTM_FOLDER)
ORTHO_FOLDER = "ORTHOPHOTOS//orthophotos"  # Path to orthophoto files
ORTHO_FOLDER = os.path.join(base_folder, ORTHO_FOLDER)
OUTPUT_FOLDER = "F://TREES//NEW/ORTHOPHOTOS" # orthophotos_deciduous" # conifers"  # Output for shadow images
OUTPUT_FOLDER = os.path.join(base_folder, OUTPUT_FOLDER)
OUTPUT_FOLDER_SHADOWS = "F:/deciduous/deciduous_shadows_renderings"  # Output for shadow images

texture_path = 'ORTHOPHOTOS//textures/deciduous.jpg' # coniferous.jpg'
texture_path = os.path.join(base_folder, texture_path)

IMG_SIZE = 224 * 2 # 224  # Image size (137x137)
NUM_VIEWS = 24  # Number of sun positions
SUN_DISTANCE = 10  # Distance of the sun from the object
SUN_HEIGHT = 5  # Sun height
GROUND_PLANE_SIZE = 100  # Ground plane size
TREE_SCALE = (1, 1, 1)  # Scale of trees (can randomize)
PLACEMENT_RATE = 0.5  # Probability of placing a tree in a lat/lon sample

# === FUNCTIONS ===
def clear_scene():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def clear_unused_data():
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

# def import_obj(obj_path):
#     bpy.ops.wm.obj_import(filepath=obj_path)
#     obj = bpy.context.selected_objects[0]
#     obj.scale = TREE_SCALE
#     assign_tree_texture(obj)
#     return obj

def import_obj(obj_path):
    # Import .obj file (may include multiple objects)
    # bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    # Collect all newly imported mesh objects
    imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if not imported_objs:
        raise RuntimeError(f"No mesh objects found in {obj_path}")
    
    # Deselect everything
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select and join all imported mesh objects
    for obj in imported_objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj  # Set one as active
    
    bpy.ops.object.join()  # Join all selected into one
    
    # Rename the result to "tree"
    merged_tree = bpy.context.active_object
    merged_tree.name = "tree"
    merged_tree.data.name = "tree"
    
    # Scale it
    merged_tree.scale = TREE_SCALE

    # Assign texture
    assign_tree_texture(merged_tree)

    return merged_tree

def import_ground_plane(size):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground_plane = bpy.context.object
    ground_plane.name = "GroundPlane"
    return ground_plane

def assign_tree_texture(tree):
    mat = bpy.data.materials.new(name="TreeTextureMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])
    tree.data.materials.clear()
    tree.data.materials.append(mat)

def setup_camera(tree):
    x, y, z = tree.location
    center = get_object_center(tree)
    highest_z = get_highest_point(tree)
    obj_size_x = max(v.co.x for v in tree.data.vertices) - min(v.co.x for v in tree.data.vertices)
    obj_size_y = max(v.co.y for v in tree.data.vertices) - min(v.co.y for v in tree.data.vertices)
    shadow_extent = max(obj_size_x, obj_size_y) * 4.75
    x = center.x + x
    y = center.y + y
    z = highest_z + z
    bpy.ops.object.camera_add(location=(x, y, z + 10))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = shadow_extent
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera
    return camera

def setup_sun():
    bpy.ops.object.light_add(type='SUN', location=(SUN_DISTANCE, 0, SUN_HEIGHT))
    return bpy.context.object


def get_tree_list():
    """Returns a sorted list of tree OBJ files in sequential order."""
    return sorted([f for f in os.listdir(OBJ_FOLDER) if f.endswith('.obj')])
    
def createTileUVMap(mesh_name):

    mesh = bpy.data.objects[mesh_name]
    bpy.ops.object.mode_set(mode='EDIT')
    
    if not mesh.data.uv_layers.active:
        bpy.ops.uv.reset()

        # bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001) # ANGLE_BASED
        
    me = mesh.data
    bm = bmesh.from_edit_mesh(me)

    min_x = min(v.co.x for v in mesh.data.vertices)
    max_x = max(v.co.x for v in mesh.data.vertices)
    min_y = min(v.co.y for v in mesh.data.vertices)
    max_y = max(v.co.y for v in mesh.data.vertices)

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

def assign_tree_texture(tree):
    """Assigns the coniferous.jpg texture as the tree material."""
    texture_path = "C:/Users/mmddd/Downloads/dataset-generation-scritps/shadows-gen/textures/coniferous.jpg"
    
    # Create material
    mat = bpy.data.materials.new(name="TreeTextureMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    # Create texture node
    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(texture_path)
    
    # Connect texture to material
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])
    
    # Clear existing materials
    tree.data.materials.clear()
    # Assign material to the tree
    tree.data.materials.append(mat)

def get_object_center(obj):
    """Finds the object's center for proper placement."""
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return sum(vertices, mathutils.Vector()) / len(vertices)


def get_lowest_point(obj):
    """Finds the lowest Z coordinate of the object for ground placement."""
    return min((obj.matrix_world @ v.co).z for v in obj.data.vertices)

def get_highest_point(obj):
    """Finds the highest Z coordinate of the object for camera placement."""
    return max((obj.matrix_world @ v.co).z for v in obj.data.vertices)

def shadow_specific_render_settings(output_path):
    # print("INSIDE")
    bpy.context.view_layer.use_pass_shadow = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    for node in nodes:
        nodes.remove(node)
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 0)
    file_output = nodes.new('CompositorNodeOutputFile')
    file_output.location = (400, 0)
    file_output.base_path = os.path.dirname(output_path)
    file_output.file_slots[0].path = os.path.splitext(os.path.basename(output_path))[0]
    file_output.format.file_format = 'PNG'
    links.new(render_layers.outputs['Shadow'], file_output.inputs[0])

    # Render and save the shadow pass
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=False)
    os.rename(os.path.join(file_output.base_path, file_output.file_slots[0].path + '0001.png'), output_path)
    print("Shadow pass rendered and saved as PNG.")

def setup_render_settings():
    bpy.context.scene.render.resolution_x = IMG_SIZE
    bpy.context.scene.render.resolution_y = IMG_SIZE
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.world.cycles_visibility.diffuse = False
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    if world.node_tree:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs[0].default_value = (1, 1, 1, 1)
    else:
        world.use_nodes = True
        bg = world.node_tree.nodes.new(type='ShaderNodeBackground')
        bg.inputs[0].default_value = (1, 1, 1, 1)
    bpy.context.scene.render.image_settings.file_format = 'PNG'

def render_shadow_views(camera, sun, output_dir, output_dir_shadows, ground_plane, tree):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_shadows, exist_ok=True)
    light_directions = []
    total_views = 10 # 30
    views_40 = total_views // 3
    views_25 = total_views // 3
    views_55 = total_views - views_40 - views_25
    for i in range(views_40):
        angle_z = i * (360 / views_40)
        x_angle = math.radians(40)
        sun.rotation_euler = (x_angle, 0, math.radians(angle_z))
        bpy.context.view_layer.update()
        light_directions.append(list(sun.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))))
        # save_shadow_render(i, output_dir, camera, output_dir_shadows)
        save_render(i, output_dir, camera, output_dir_shadows)
        # save_shadowPoints(i, output_dir, camera, output_dir_shadows, ground_plane, sun)
    for i in range(views_40, views_40 + views_55):
        angle_z = (i - views_40) * (360 / views_55)
        x_angle = math.radians(55)
        sun.rotation_euler = (x_angle, 0, math.radians(angle_z))
        bpy.context.view_layer.update()
        light_directions.append(list(sun.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))))
        # save_shadow_render(i, output_dir, camera, output_dir_shadows)
        save_render(i, output_dir, camera, output_dir_shadows)
        # save_shadowPoints(i, output_dir, camera, output_dir_shadows, ground_plane, sun)
    for i in range(views_40 + views_55, total_views):
        angle_z = (i - views_40 - views_55) * (360 / views_25)
        x_angle = math.radians(25)
        sun.rotation_euler = (x_angle, 0, math.radians(angle_z))
        bpy.context.view_layer.update()
        light_directions.append(list(sun.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))))
        # save_shadow_render(i, output_dir, camera, output_dir_shadows)
        save_render(i, output_dir, camera, output_dir_shadows)
        # save_shadowPoints(i, output_dir, camera, output_dir_shadows, ground_plane, sun)
    light_directions_path = os.path.join(output_dir, "light_directions.txt")
    with open(light_directions_path, 'w') as f:
        for direction in light_directions:
            f.write(f"{direction[0]} {direction[1]} {direction[2]}/n")
    print("📝 Light directions saved to", light_directions_path)

def save_render(index, output_dir, camera, output_dir_shadows):
    # ** 
    """Renders and saves the image and shadow mask."""
    set_white_background()

    file_path = os.path.join(output_dir, f"view_{index:03d}.png")
    bpy.context.scene.render.filepath = file_path
    bpy.ops.render.render(write_still=True)
    print(f"📸 Rendered view {index:03d} with shadow detection")

# === DTM specific ===
def import_dtm(dtm_path):
    """Imports a DTM (terrain) as a mesh."""
    bpy.ops.importgis.georaster(filepath=dtm_path, importMode='DEM_RAW')
    dtm = bpy.context.selected_objects[0]
    dtm.name = "DTM"
    return dtm

def apply_textures_again(dtm, tree):
    """Applies an orthophoto as a texture to the DTM."""
    # dtm.data.materials.clear()
    dtm.active_material = bpy.data.materials.get("OrthoMaterial")

    # tree.data.materials.clear()
    assign_tree_texture(tree)

def set_specular_to_zero(material):
    if material and material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                if "Specular" in node.inputs:
                    node.inputs["Specular"].default_value = 0.0
                if "Roughness" in node.inputs:
                    node.inputs["Roughness"].default_value = 0.0

def apply_ortho_texture(dtm, ortho_path):
    """Applies an orthophoto as a texture to the DTM."""
    createTileUVMap(dtm.name)

    mat = bpy.data.materials.new(name="OrthoMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    # Apply this to all materials if needed
    for mat in bpy.data.materials:
        set_specular_to_zero(mat)
    
    # Create texture node
    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(ortho_path)
    
    # Connect texture to material
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])
    
    dtm.data.materials.append(mat)

    dtm.visible_shadow = False

def get_dtm_and_ortho():
    """Returns a dictionary of {DTM: Orthophoto} pairs."""
    dtm_files = sorted([f for f in os.listdir(DTM_FOLDER) if f.endswith('.tif')])
    dtms_with_orthos = {}

    for dtm_file in dtm_files:
        ortho_file = dtm_file.replace('dtm', 'ortho').replace(".tif", ".png")
        dtm_path = os.path.join(DTM_FOLDER, dtm_file)
        ortho_path = os.path.join(ORTHO_FOLDER, ortho_file)

        print('File path of dtm_files:', DTM_FOLDER)
        print('File path of ortho_files:', ortho_path)

        if os.path.exists(ortho_path):
            dtms_with_orthos[dtm_path] = ortho_path
        else:
            print(f"⚠️ Warning: No orthophoto found for {dtm_file}")

    return dtms_with_orthos

def calculate_dtm_ranges(dtm_pairs, total_trees):
    dtm_ranges = []
    trees_per_dtm = total_trees // len(dtm_pairs)
    start_idx = 0

    for dtm, ortho in dtm_pairs:
        end_idx = start_idx + trees_per_dtm
        dtm_ranges.append((start_idx, end_idx, dtm, ortho))
        start_idx = end_idx

    # Handle any remaining trees
    if start_idx < total_trees:
        dtm_ranges[-1] = (dtm_ranges[-1][0], total_trees, dtm_ranges[-1][2], dtm_ranges[-1][3])

    return dtm_ranges

def get_dtm_for_tree(tree_name, dtm_ranges):
    """Quickly finds the corresponding DTM and orthophoto based on the tree number."""
    tree_number = int(tree_name.split('_')[1].split('.')[0])
    
    for start_idx, end_idx, dtm, ortho in dtm_ranges:
        if start_idx <= tree_number < end_idx:
            return dtm, ortho
    
    return dtm_ranges[-1][2], dtm_ranges[-1][3]  # Default to the last DTM if out of range

def clear_scene_except_dtm():
    """Frees up memory by removing unused meshes, materials, and images."""
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'LIGHT', 'CAMERA'} and "DTM" not in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in bpy.data.meshes:
        if mesh.users == 0  and "DTM" not in mesh.name:
            bpy.data.meshes.remove(mesh)
    
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)

    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)

    bpy.ops.outliner.orphans_purge(do_recursive=True)

def apply_smooth_shading(obj):
    """Applies smooth shading to the given object."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
# ----------------------------------------------

def get_random_xy_on_dtm(dtm_obj, margin=5.0):
    """Returns a random (x, y) within the bounding box of the DTM mesh, with an optional margin."""
    vertices = [v.co for v in dtm_obj.data.vertices]
    min_x = min(v.x for v in vertices) + margin
    max_x = max(v.x for v in vertices) - margin
    min_y = min(v.y for v in vertices) + margin
    max_y = max(v.y for v in vertices) - margin

    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    return x, y
def get_dtm_height_at_point(dtm_obj, x, y):
    """Raycasts from above to find the Z height of the DTM at (x, y)."""
    origin = mathutils.Vector((x, y, 1000))  # Ray origin far above terrain
    direction = mathutils.Vector((0, 0, -1))  # Pointing down
    result, location, normal, index = dtm_obj.ray_cast(origin, direction)
    return location.z if result else None

def main(): # start, end):
    dtm_pairs = list(get_dtm_and_ortho().items())
    tree_files = get_tree_list()
    # total_trees = len(tree_files)
    total_trees = len(tree_files) # min(len(tree_files), 4000)  # Limit to 4000 trees

    print(f"🌲 Found {total_trees} tree OBJ files and {len(dtm_pairs)} DTM files.")

    if not tree_files or not dtm_pairs:
        raise ValueError("❌ No tree OBJ files or DTM files found!")

    dtm_ranges = calculate_dtm_ranges(dtm_pairs, total_trees)

    print('dtm_ranges:', dtm_ranges)

    current_dtm = None

    for tree_file in tree_files: # [:7000]:
        tree_number = int(tree_file.split('_')[1].split('.')[0])
        # if tree_number < start:
        #     continue
        # if tree_number > end:
        #     continue
        output_dir = os.path.join(OUTPUT_FOLDER, f"tree_{tree_number:04d}", "rendering")
        output_dir_shadows = os.path.join(OUTPUT_FOLDER_SHADOWS, f"tree_{tree_number:04d}", "rendering")
        if os.path.exists(output_dir): ##
            print(f"🚀 Tree {tree_number} already processed. Skipping...")
            continue
        print(f"🌍 Processing tree {tree_file}")
        # clear_scene()
        # clear_unused_data()
        
        dtm_path, ortho_path = get_dtm_for_tree(tree_file, dtm_ranges)
        
        if dtm_path != current_dtm:
            print(f"🌍 Switching to DTM for tree {tree_file}: {dtm_path}")
            clear_scene()
            dtm = import_dtm(dtm_path)
            apply_ortho_texture(dtm, ortho_path)
            current_dtm = dtm_path
        else:
            print(f"🌍 Continuing with current DTM for tree {tree_file}")
            clear_scene_except_dtm()
        
        tree_path = os.path.join(OBJ_FOLDER, tree_file)
        tree = import_obj(tree_path)

        tree.location = (0, 0, 0)

        #
        apply_textures_again(dtm, tree)

        apply_smooth_shading(dtm)
        bpy.context.view_layer.objects.active = dtm
        bpy.ops.object.select_all(action='DESELECT')
        dtm.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        dtm.location = (0, 0, 0)

        dtm_object = bpy.data.objects.get("DTM")
        if dtm_object:
            dtm.visible_shadow = False  # Prevent it from casting shadows
            dtm.display_type = 'SOLID'  # Optional: make it solid in viewport

        # 
        placed = False
        attempts = 0
        while not placed and attempts < 10:
            x, y = get_random_xy_on_dtm(dtm)
            z = get_dtm_height_at_point(dtm, x, y)
            if z is not None:
                tree.location = (x, y, z)
                placed = True
            attempts += 1
        tree.location = (x, y, z) if placed else (0, 0, 0)

        camera = setup_camera(tree)
        sun = setup_sun()

        dtm.hide_render = False
        setup_render_settings()
        render_shadow_views(camera, sun, output_dir, output_dir_shadows, dtm, tree)

        #########
        side_output_dir = output_dir.replace("rendering", "side_views")
        print('Side output directory:', side_output_dir)
        distance = estimate_safe_camera_distance(tree)
        # Set tree origin to geometry before side views
        bpy.context.view_layer.objects.active = tree
        bpy.ops.object.select_all(action='DESELECT')
        tree.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        tree.location = (0, 0, 0)  # Reset tree location for side views
        render_side_views(tree, dtm, side_output_dir, num_views=8, distance=distance)

        # return

    print("✅ All DTMs and trees processed.")

#############

def setup_fixed_camera(tree, distance=None, elevation_above=1):
    """Sets up a fixed camera in front of the tree that will remain stationary."""
    center = get_object_center(tree)
    
    # Calculate camera distance based on tree size if not provided
    if distance is None:
        # Get tree bounding box dimensions
        bbox = tree.bound_box
        tree_width = max(
            abs(bbox[0][0] - bbox[4][0]),  # X dimension
            abs(bbox[0][1] - bbox[2][1])   # Y dimension
        )
        tree_height = abs(bbox[0][2] - bbox[4][2])  # Z dimension
        
        # Calculate distance to fit the entire tree in view
        # Using field of view and tree dimensions to determine optimal distance
        max_dimension = max(tree_width, tree_height)
        # Assume camera FOV of ~50 degrees, add margin for safety
        distance = max_dimension #* 1.5  # 1.5x for margin
        
        print(f"📏 Tree dimensions: {tree_width:.2f} x {tree_height:.2f}, calculated distance: {distance:.2f}")
    
    # Position camera in front of tree (along positive Y axis)
    x = center.x
    y = center.y + distance  # Distance based on tree size
    z = center.z + elevation_above  # Slightly above tree center
    
    # Add and aim camera at tree center
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    camera.name = "FixedSideCamera"
    
    # Point camera at tree center
    direction = center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    bpy.context.scene.camera = camera
    return camera
def render_side_views(tree, dtm, output_dir, num_views=8, distance=10):
    """Render side views by rotating the tree while keeping camera fixed."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up fixed camera once (distance calculated automatically based on tree size)
    camera = setup_fixed_camera(tree)
    
    if camera is None:
        print("⚠️ Failed to setup fixed camera.")
        return
    
    # Store original tree rotation
    original_rotation = tree.rotation_euler.copy()
    
    # Define rotation angles (in degrees)
    rotation_angles = [i * (360 / num_views) for i in range(num_views)]
    
    # Save rotation angles to separate text file (degrees only)
    rotation_degrees_path = os.path.join(output_dir, "rotation_degrees.txt")
    with open(rotation_degrees_path, 'w') as f:
        for angle in rotation_angles:
            f.write(f"{angle:.2f}\n")
    print(f"📝 Rotation degrees saved to {rotation_degrees_path}")
    
    for i, angle_deg in enumerate(rotation_angles):
        # Rotate tree around Z-axis (vertical)
        tree.rotation_euler = (
            original_rotation[0],  # Keep X rotation
            original_rotation[1],  # Keep Y rotation  
            original_rotation[2] + math.radians(angle_deg)  # Rotate around Z
        )
        
        # Update scene to apply rotation
        bpy.context.view_layer.update()
        
        # Update camera to always look at the tree center after rotation
        tree_center = get_object_center(tree)
        direction = tree_center - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        file_path = os.path.join(output_dir, f"side_view_{i:03d}.png")
        pose_path = os.path.join(output_dir, f"side_view_{i:03d}_pose.txt")

        # Make DTM object invisible in render
        dtm_name = "DTM"
        if dtm_name in bpy.data.objects:
            dtm_obj = bpy.data.objects[dtm_name]
            dtm_obj.hide_render = True  # Hide from final render
            dtm_obj.hide_viewport = False  # Optional: still visible in viewport
            
        # Set transparent background
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Ensure alpha channel
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # Use format that supports transparency
                
        # Render
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

        print(f"📸 Saved side view {i} at rotation {angle_deg}°.")
    
    # Restore original tree rotation
    tree.rotation_euler = original_rotation
    bpy.context.view_layer.update()

def estimate_safe_camera_distance(tree, margin=40):
    bbox = tree.bound_box
    radius = max(
        abs(bbox[0][0] - bbox[4][0]),
        abs(bbox[0][1] - bbox[2][1])
    ) / 2
    return radius + margin

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Orthophotos Generation Script")
    parser.add_argument('--start', type=int, required=True, help='Start index for tree processing')
    parser.add_argument('--end', type=int, required=True, help='End index for tree processing')
    args, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
    return args

if __name__ == "__main__":
    # args = parse_args()
    main() # args.start, args.end)

# blender --python 4_orthophotos-gen.py 


## scp -P 31415 F:\TREES\NEW\ORTHOPHOTOS_IMG.tar grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/DATASET