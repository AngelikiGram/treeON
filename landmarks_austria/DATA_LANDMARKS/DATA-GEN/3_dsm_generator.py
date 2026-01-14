import bpy
import os
import numpy as np
import time

MAX_TIME = 100  # seconds

# Settings
resolution = 3 # 1  # Smaller grid cells (higher resolution)
OBJ_FOLDER = "F:/TREES/NEW/simplified" # deciduous" # conifers"
OUTPUT_FOLDER = "F:/TREES/NEW/DSM_OBJ" # dsms_deciduous" # conifers" 

TARGET_VERTEX_COUNT = 1000 # 6000 # 300 # 6000 # 3000
TREE_SCALE = (1, 1, 1)

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_obj(obj_path):
    """Import an OBJ file, handling errors if needed."""
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj = bpy.context.selected_objects[0]
        # obj.scale = TREE_SCALE
        # obj.location = (0, 0, 0)
        return obj
    except Exception as e:
        print(f"❌ Error importing {obj_path}: {e}")
        return None

def get_tree_list():
    """Returns a sorted list of tree OBJ files."""
    return sorted([f for f in os.listdir(OBJ_FOLDER) if f.endswith('.obj')])

def generate_dsm(tree, output_dsm_path, resolution):
    """Generate a DSM with a higher-resolution grid instead of using subdivision."""

    if tree is None:
        print("⚠️ Skipping due to import error.")
        return

    # Get mesh vertices in world space
    vertices = np.array([tree.matrix_world @ v.co for v in tree.data.vertices])

    # Determine DSM grid bounds
    min_x, min_y = np.min(vertices[:, :2], axis=0)
    max_x, max_y = np.max(vertices[:, :2], axis=0)
    
    # Adjust resolution to get close to TARGET_VERTEX_COUNT
    grid_width = int((max_x - min_x) / resolution) + 1
    grid_height = int((max_y - min_y) / resolution) + 1

    total_vertices = grid_width * grid_height
    print(f"ℹ️ Initial grid size: {grid_width}x{grid_height} = {total_vertices} vertices")


    # Initialize DSM grid with 0 height (flat ground)
    dsm = np.zeros((grid_height, grid_width))

    # Assign height values based on the highest vertex in each grid cell
    for v in vertices:
        grid_x = int((v[0] - min_x) / resolution)
        grid_y = int((v[1] - min_y) / resolution)
        dsm[grid_y, grid_x] = max(dsm[grid_y, grid_x], v[2])

    # Create a new mesh for the DSM
    mesh = bpy.data.meshes.new("DSM_Mesh")
    dsm_obj = bpy.data.objects.new("DSM_Object", mesh)
    bpy.context.collection.objects.link(dsm_obj)

    # Create vertices (DSM surface + ground base)
    verts = []
    for y in range(grid_height):
        for x in range(grid_width):
            verts.append((min_x + x * resolution, min_y + y * resolution, dsm[y, x]))  # DSM surface

    base_offset = len(verts)  # Ground base starts here
    for y in range(grid_height):
        for x in range(grid_width):
            verts.append((min_x + x * resolution, min_y + y * resolution, 0))  # Ground base

    # Create faces
    faces = []

    # Top surface faces (DSM mesh)
    for y in range(grid_height - 1):
        for x in range(grid_width - 1):
            idx = y * grid_width + x
            faces.append([idx, idx + 1, idx + grid_width + 1, idx + grid_width])

    # Side faces (connecting DSM to the ground)
    for y in range(grid_height - 1):
        for x in range(grid_width - 1):
            idx = y * grid_width + x
            ground_idx = base_offset + idx  # Corresponding ground vertex
            
            faces.append([idx, ground_idx, ground_idx + 1, idx + 1])  # Front face
            faces.append([idx, idx + grid_width, ground_idx + grid_width, ground_idx])  # Side face

    # Bottom face (closing the base)
    for y in range(grid_height - 1):
        for x in range(grid_width - 1):
            idx = base_offset + y * grid_width + x
            faces.append([idx, idx + 1, idx + grid_width + 1, idx + grid_width])

    # # Create vertices
    # verts = [(min_x + x * resolution, min_y + y * resolution, dsm[y, x]) 
    #          for y in range(grid_height) for x in range(grid_width)]

    # # Create faces
    # faces = []
    # for y in range(grid_height - 1):
    #     for x in range(grid_width - 1):
    #         idx = y * grid_width + x
    #         faces.append([idx, idx + 1, idx + grid_width + 1, idx + grid_width])

    # Assign geometry to mesh
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    print(f"🔹 Final DSM mesh has {len(verts)} vertices.")

    # save blend file
    bpy.ops.wm.save_as_mainfile(filepath='F:/conifers/dsm.blend')

    # Remove the tree object
    bpy.ops.object.select_all(action='DESELECT')
    tree.select_set(True)
    bpy.ops.object.delete()

    # subdivide the vertices of mesh to get close to TARGET_VERTEX_COUNT
    start_time = time.time()
    while len(verts) < TARGET_VERTEX_COUNT:
        if time.time() - start_time > MAX_TIME:
            print("⏰ Stopping: time limit reached.")
            break
        bpy.ops.object.select_all(action='DESELECT')
        dsm_obj.select_set(True)
        bpy.context.view_layer.objects.active = dsm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
        verts = [(v.co[0], v.co[1], v.co[2]) for v in dsm_obj.data.vertices]

    # Export as OBJ
    bpy.ops.wm.obj_export(filepath=output_dsm_path)
    print(f"✅ DSM exported to {output_dsm_path}")
    
def main(): # start, end):
    """Process tree OBJ files and generate DSMs."""
    tree_files = get_tree_list()
    print(f"🌲 Found {len(tree_files)} tree OBJ files.")
    if not tree_files:
        raise ValueError("❌ No tree OBJ files found!")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for tree_file in tree_files: # [start:end]:
        tree_number = int(tree_file.split('_')[1].split('.')[0])
        # if tree_number < 1000:
        #     continue
        output_dsm_path = os.path.join(OUTPUT_FOLDER, f"tree_{tree_number:04d}.obj")

        if os.path.exists(output_dsm_path):
            print(f"🚀 DSM for Tree {tree_number} already exists. Skipping...")
            continue

        print(f"🌍 Processing Tree {tree_file} -> Generating DSM")
        clear_scene()

        # Import tree
        tree_path = os.path.join(OBJ_FOLDER, tree_file)
        tree = import_obj(tree_path)

        # Generate and save DSM
        generate_dsm(tree, output_dsm_path, resolution)

    print("All DSMs processed.")

import argparse
def parse_args():
    """Parse command-line arguments for start and end indices."""
    parser = argparse.ArgumentParser(description="DSM Generation Script")
    parser.add_argument('--start', type=int, required=True, help='Start index for tree processing')
    parser.add_argument('--end', type=int, required=True, help='End index for tree processing')
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    # args = parse_args()
    main() # args.start, args.end)

# blender --python 3_dsm_generator.py