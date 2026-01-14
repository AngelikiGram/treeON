import os
import sys
import torch
import numpy as np
import visdom
import torchvision.transforms as transforms

sys.path.append('./auxiliary/')
from query_points_gen import *
from occupancy_compute import extract_top_k_occupied_points
from scipy.io import loadmat
from PIL import Image

def save_points_as_ply(points, filename, colors=None):
    """
    Saves the given points (tensor) as a PLY file.
    points: (N, 3) torch.Tensor or ndarray.
    colors: (N, 3) torch.Tensor or ndarray, values in [0, 255] or [0, 1].
    filename: output path (.ply).
    """
    # Normalize input
    if isinstance(points, list):
        if len(points) == 1 and isinstance(points[0], torch.Tensor):
            points = points[0]  # unwrap
        else:
            points = np.array([
                p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                for p in points
            ])
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # Ensure points are (N,3)
    points = np.asarray(points).reshape(-1, 3)

    print(f"Saving {points.shape[0]} points to {filename}")
    N = points.shape[0]

    # Prepare colors
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        colors = np.asarray(colors).reshape(-1, 3)

        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Write PLY
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {N}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

        for i in range(N):
            x, y, z = points[i].tolist()
            if colors is not None:
                r, g, b = colors[i].tolist()
                f.write(f'{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n')
            else:
                f.write(f'{x:.8f} {y:.8f} {z:.8f}\n')

def load_dsm_mat(mat_path):
    mat = loadmat(mat_path)
    # Assume DSM is stored under key 'dsm' or similar
    for key in mat:
        if key.lower().startswith('vertices'):
            dsm = mat[key]
            break
    else:
        raise ValueError(f"DSM not found in {mat_path}")
    dsm = torch.tensor(dsm, dtype=torch.float32)
    return dsm

def load_ortho_img(img_path):
    img = Image.open(img_path).convert('RGB')
    rgb_transforms = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = rgb_transforms(img)
    img = img.unsqueeze(0)  # (1, 3, 224, 224)
    return img

def main(args):
    viz = visdom.Visdom(env=args.env, port=8099)
    viz.close()

    sys.path.append('./auxiliary/models/')
    sys.path.append('./auxiliary/dataset')
    # Import model
    if args.model == '1':
        from model_normal_categories import TreeReconstructionNet
    elif args.model == '2':
        from model_dsm import TreeReconstructionNet
    elif args.model == '3':
        from model_ortho import TreeReconstructionNet
    elif args.model == '4':
        from model_normal_with_noise import TreeReconstructionNet
    elif args.model == '10':
        from model_normal_categories_noRefinement import TreeReconstructionNet
    else:
        from model_colors_fig import TreeReconstructionNet
    # Load DSM and ortho
    dsm = load_dsm_mat(args.dsm_mat)
    # Normalize DSM points to unit cube [0,1]^3 using shared scale
    def normalize_to_unit_cube(points):
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord) / 2.0
        points_centered = points - center
        extent = (max_coord - min_coord).max()
        scale = extent + 1e-6
        normalized = points_centered / scale + 0.5
        return normalized, center, scale
    dsm_np = dsm.detach().cpu().numpy() if isinstance(dsm, torch.Tensor) else np.asarray(dsm)
    dsm_np, dsm_center, dsm_scale = normalize_to_unit_cube(dsm_np)
    dsm = torch.tensor(dsm_np, dtype=torch.float32)
    # Ensure DSM is (1, N, 3) for PointNet
    if dsm.ndim == 2 and dsm.shape[1] == 3:
        dsm = dsm.unsqueeze(0)
    elif dsm.ndim == 1 and dsm.shape[0] == 3:
        dsm = dsm.view(1, 1, 3)
    ortho = load_ortho_img(args.ortho_img)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 1
    if args.variable == '1':
        query_points = create_query_points(B, dsm, num_query_points=args.num_query_points).to(device)
    elif args.variable == '2':
        query_points = sample_points_in_convex_hull(B, dsm, num_query_points=args.num_query_points)
    elif args.variable == '3':
        query_points = create_normalized_query_points(B, num_query_points=args.num_query_points)
    model = TreeReconstructionNet(num_points=args.num_points, num_species=14).to(device)
    dsm = dsm.to(device)
    ortho = ortho.to(device)
    query_points = query_points.to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        output_occ, _, pred_colors = model(dsm, ortho, query_points)
        probs = torch.sigmoid(output_occ)
        pred_points, topk_indices = extract_top_k_occupied_points(probs, query_points, args.num_points)
        
        # Denormalize prediction points back to original coordinate system
        def denormalize_from_unit_cube(points, center, scale):
            points_shifted = points - 0.5  # shift from [0,1] to [-0.5,0.5]
            points_scaled = points_shifted * scale  # scale back
            points_final = points_scaled + center  # translate back
            return points_final
        
        # Denormalize predicted points
        if isinstance(pred_points, torch.Tensor):
            pred_points_np = pred_points.detach().cpu().numpy()
        elif isinstance(pred_points, list):
            # Handle list of tensors
            pred_points_np = np.array([p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in pred_points])
        else:
            pred_points_np = np.array(pred_points)
        pred_points_denorm = denormalize_from_unit_cube(pred_points_np, dsm_center, dsm_scale)
        pred_points = torch.tensor(pred_points_denorm, dtype=torch.float32)
    # Extract DSM id from filename (e.g. dsm_1.mat -> 1)
    import re
    dsm_basename = os.path.basename(args.dsm_mat)
    match = re.search(r'dsm_(\d+)', dsm_basename)
    if match:
        dsm_id = match.group(1)
    else:
        dsm_id = 'unknown'
    output_path = os.path.join(args.output_root, f"output_{dsm_id}.ply")
    save_points_as_ply(pred_points, output_path, colors=pred_colors)
    print(f"Saved: {output_path}")

    # Also save query points as PLY (denormalized)
    query_np = query_points.detach().cpu().numpy().reshape(-1, 3)
    query_denorm = denormalize_from_unit_cube(query_np, dsm_center, dsm_scale)
    query_ply_path = os.path.join(args.output_root, f"query_points.ply")
    save_points_as_ply(query_denorm, query_ply_path)
    print(f"Saved query points: {query_ply_path}")

    # Plot DSM points (denormalized for visualization)
    dsm_np_norm = dsm.detach().cpu().numpy().reshape(-1, 3)  # normalized DSM
    dsm_np = denormalize_from_unit_cube(dsm_np_norm, dsm_center, dsm_scale)  # denormalized DSM
    dsm_color = np.tile([0, 1, 0], (len(dsm_np), 1))
    viz.scatter(
        X=dsm_np,
        opts=dict(
            markersize=3,
            markercolor=dsm_color,
            title='DSM Points',
            xlabel='X', ylabel='Y', zlabel='Z',
            dim=3
        )
    )

    # Plot generated points (already denormalized)
    if isinstance(pred_points, torch.Tensor):
        gen_np = pred_points.detach().cpu().numpy().reshape(-1, 3)
    elif isinstance(pred_points, list) and len(pred_points) > 0 and isinstance(pred_points[0], torch.Tensor):
        gen_np = np.stack([p.detach().cpu().numpy() for p in pred_points]).reshape(-1, 3)
    else:
        gen_np = np.asarray(pred_points).reshape(-1, 3)
    gen_color = np.tile([1, 0, 0], (len(gen_np), 1))
    viz.scatter(
        X=gen_np,
        opts=dict(
            markersize=3,
            markercolor=gen_color,
            title='Generated Points',
            xlabel='X', ylabel='Y', zlabel='Z',
            dim=3
        )
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate output point cloud from DSM .mat and orthophoto image.")
    parser.add_argument("--dsm_mat", type=str, required=True, help="Path to DSM .mat file")
    parser.add_argument("--ortho_img", type=str, required=True, help="Path to orthophoto image")
    parser.add_argument("--output_root", type=str, required=True, help="Path to output folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model", type=str, default="5", help="Model type")
    parser.add_argument("--num_points", type=int, default=4500, help="Number of output points")
    parser.add_argument("--env", type=str, default="p2_val", help="Visdom environment name")
    parser.add_argument("--variable", type=str, default="3", help="Query points generation method")
    parser.add_argument("--num_query_points", type=int, default=51000, help="Number of query points")
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=1 python validation/gen_point_cloud/generate_output_pointcloud_from_files.py --dsm_mat france_data/DSM_MAT/dsm_15.mat --ortho_img france_data/ORTHOPHOTOS/ortho_15.png --output_root results/landmarks/FRANCE --model_path log/mixed_all_450/network.pth --env test_env


# mkdir results/landmarks/FRANCE