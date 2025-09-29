import os
import trimesh
import scipy.io
import numpy as np

def load_obj_all_vertices(obj_path):
    vertices = []
    colors = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                coords = list(map(float, parts[1:4]))
                vertices.append(coords)
                if len(parts) >= 7:  # RGB included
                    r, g, b = map(float, parts[4:7])
                    # Normalize if necessary
                    if max(r, g, b) > 1.0:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    colors.append([r, g, b])
                else:
                    colors.append([1.0, 1.0, 1.0])  # default white
    return np.array(vertices), np.array(colors)

def convert_obj_to_mat_DSM(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            obj_path = os.path.join(input_folder, filename)
            mat_filename = os.path.splitext(filename)[0] + ".mat"
            mat_path = os.path.join(output_folder, mat_filename)

            if os.path.exists(mat_path):
                print(f"Skipping existing file: {mat_filename}")
          #      continue

            try:
                mesh = trimesh.load(obj_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

                all_vertices, all_colors = load_obj_all_vertices(obj_path)

                print(f"File: {filename} | Vertices: {len(all_vertices)} | Colors: {len(all_colors)}")
# | Faces: {len(mesh.faces)} | 
                data = {
                    "vertices": all_vertices,
                   # "faces": mesh.faces,
                    "colors": all_colors
                }
                scipy.io.savemat(mat_path, data)
                print(f"Converted: {filename} to {mat_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def convert_obj_to_mat(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):

            # Check if the filename matches the expected format
            parts = filename.split("_")
            if len(parts) != 2 or not parts[1].endswith(".obj"):
                print(f"Skipping invalid filename: {filename}")
                continue

            # Extract the tree number
            try:
                tree_number = int(parts[1].split(".")[0])
            except ValueError:
                print(f"Skipping invalid filename: {filename}")
               # continue

            # # Convert only if the tree number is 4156 or higher
            # if tree_number < 6580:
            #     continue

            
            # Output .mat file path
            mat_filename = os.path.splitext(filename)[0] + ".mat"
            mat_filename = mat_filename.replace("dsm_", "")
            print(f"mat_filename: {mat_filename}")
            mat_path = os.path.join(output_folder, mat_filename)

            if os.path.exists(mat_path):
                print(f"Skipping existing file: {mat_filename}")
                continue

            # Full path to the .obj file
            obj_path = os.path.join(input_folder, filename)
            
            try:
                # Load the .obj file using trimesh
                mesh = trimesh.load(obj_path)

                # If it's a Scene, try extracting the first mesh
                if isinstance(mesh, trimesh.Scene):
                    if len(mesh.geometry) == 0:
                        print(f"Skipping empty scene: {filename}")
                      #  continue
                    # Get the first mesh in the scene
                    mesh = list(mesh.geometry.values())[0]

                # Prepare the data for saving
                data = {
                    'vertices': mesh.vertices,
                  #  'faces': mesh.faces
                }

                # Save data to .mat file
                scipy.io.savemat(mat_path, data)

                print(f"Converted: {filename} to {mat_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

input_folder = "./DSM_OBJ" # _NORMALIZED_ALIGNED" # SPLIT_TREES2" 
output_folder = "./DSM" # TREES
# input_folder = "D:/TREES_DATASET/_SPRUCES1/COLORED_TREES" # DSM_OBJ_NORMALIZED-innerPoints"
# output_folder = "D:/TREES_DATASET/_SPRUCES1/TREES_MAT"
os.makedirs(output_folder, exist_ok=True)
convert_obj_to_mat_DSM(input_folder, output_folder)
