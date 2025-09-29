import scipy.io
import numpy as np

def mat_to_obj(mat_path, obj_path, var_name='vertices'):
    # Load .mat file
    data = scipy.io.loadmat(mat_path)
    
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in {mat_path}. Keys: {list(data.keys())}")

    points = data[var_name]  # [N, 3]

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected shape [N, 3], got {points.shape}")

    # Write .obj file
    with open(obj_path, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    print(f"Saved {points.shape[0]} points to {obj_path}")

# Example usage:
input = 'F://TREES//DATASET//DSM//tree_1500.mat'
output = 'tree_1500.obj'
mat_to_obj(input, output, var_name='vertices')