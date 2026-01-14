import os
import trimesh
import scipy.io
import numpy as np

def load_obj_all_vertices(obj_path):
    vertices = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):  # Extracts all vertex lines
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def convert_obj_to_mat_DSM(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            obj_path = os.path.join(input_folder, filename)
            mat_filename = os.path.splitext(filename)[0] + ".mat"
            mat_path = os.path.join(output_folder, mat_filename)

            # if os.path.exists(mat_path):
            #     print(f"Skipping existing file: {mat_filename}")
            #     continue

            try:
                # Load mesh without modifying topology
                mesh = trimesh.load(obj_path, process=False)

                # Merge all objects if it's a scene
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

                # Extract vertices directly from file
                all_vertices = load_obj_all_vertices(obj_path)

                print(f"File: {filename} | All Vertices: {len(all_vertices)} | Trimesh Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)}")

                # Save all extracted data
                data = {
                    "vertices": all_vertices,  # All vertices in the file
                    # "vertices": mesh.vertices,  # Only used vertices
                    "faces": mesh.faces
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
               # continue

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
                    'faces': mesh.faces
                }

                # Save data to .mat file
                scipy.io.savemat(mat_path, data)

                print(f"Converted: {filename} to {mat_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Specify input and output folders
input_folder = "F:/conifers/.OBJ CONIFERS/conifers_simplified_6000" # DSM/dsms_conifers_sampled2/" # dsms_conifers_sampled/" # conifers_detailed/" # conifers_simplified/"
output_folder = "F:/conifers/.OBJ CONIFERS/conifers_mat_6000" # DSM/conifers_dsms_mat_6000/" # conifers_mat_detailed/"
# Specify input and output folders
# input_folder = "C:/Users/mmddd/Documents/Tree-Generation-from-Single-Orthophoto-and-DSM/testing" # DSM/dsms_conifers_sampled2/" # dsms_conifers_sampled/" # conifers_detailed/" # conifers_simplified/"
# output_folder = "C:/Users/mmddd/Documents/Tree-Generation-from-Single-Orthophoto-and-DSM/testing" # DSM/conifers_dsms_mat_6000/" # conifers_mat_detailed/"
input_folder = "F:/conifers/DSM/dsms_innerPoints-3000/"
output_folder = "F:/conifers/DSM/dsms_innerPoints_mat-3000/"

input_folder = "G:/deciduous/dsms_deciduous-sampled/"
output_folder = "G:/deciduous/dsms_deciduous-mat/"

input_folder = "C:/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/DSM_OBJ/"
output_folder = "C:/Users/mmddd/Documents/p2-tree-gen/validation/landmarks_austria/DSM/"
convert_obj_to_mat_DSM(input_folder, output_folder)