import torch
from torch.utils.data import DataLoader
import argparse
import trimesh
import os
import numpy as np
import sys

sys.path.append('./auxiliary/')
from query_points_gen import *
from occupancy_compute import extract_top_k_occupied_points, compute_occupancy, extract_threshold_occupied_points, extract_top_k_occupied_points_val
sys.path.append('./auxiliary/dataset')
from dataset_test_landmarks import TreeDataset

def save_points_as_ply(points, filename):
    """
    Saves the given points (tensor) as a PLY file.
    points: (N, 3) tensor.
    filename: output path (.ply).
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    cloud = trimesh.PointCloud(points)
    cloud.export(filename)

def collate_fn(batch):
    imgs, initial_vertices, index_views, filenames, center_dsm, scale_dsm, original_dsm = zip(*batch)

    imgs = torch.stack([torch.tensor(img, dtype=torch.float32) if isinstance(img, np.ndarray) else img for img in imgs])
    
    # Handle variable-size point clouds by padding to max size
    max_vertices = max(len(v) for v in initial_vertices)
    max_dsm = max(len(d) for d in original_dsm)
    
    # Pad initial_vertices to max_vertices
    padded_initial_vertices = []
    for v in initial_vertices:
        v_tensor = torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v
        if len(v_tensor) < max_vertices:
            # Pad with zeros or repeat last point
            padding = torch.zeros(max_vertices - len(v_tensor), v_tensor.shape[1])
            v_tensor = torch.cat([v_tensor, padding], dim=0)
        padded_initial_vertices.append(v_tensor)
    
    # Pad original_dsm to max_dsm
    padded_original_dsm = []
    for d in original_dsm:
        d_tensor = torch.tensor(d, dtype=torch.float32) if isinstance(d, np.ndarray) else d
        if len(d_tensor) < max_dsm:
            # Pad with zeros or repeat last point
            padding = torch.zeros(max_dsm - len(d_tensor), d_tensor.shape[1])
            d_tensor = torch.cat([d_tensor, padding], dim=0)
        padded_original_dsm.append(d_tensor)
    
    initial_vertices = torch.stack(padded_initial_vertices)
    original_dsm = torch.stack(padded_original_dsm)

    index_views = torch.tensor(index_views)

    filename = [f for f in filenames]

    center_dsm = torch.stack([
        torch.tensor(c, dtype=torch.float32) if not torch.is_tensor(c) else c
        for c in center_dsm
    ])

    scale_dsm = torch.stack([
        torch.tensor([s], dtype=torch.float32) if not torch.is_tensor(s) else s
        for s in scale_dsm
    ]).squeeze(1)  # Squeeze to make shape [B] instead of [B, 1]

    return imgs, initial_vertices, index_views, filename, center_dsm, scale_dsm, original_dsm

import visdom

def normalize_dsm_to_unit_top(dsm_points):
    """
    Normalize each DSM point cloud so that:
    - The peak height (max y) is at 1
    - The center is at (0, 0) in x and z axes
    - Isotropic scaling based on the y-range (height)
    Input: dsm_points of shape (B, N, 3) [x, y, z] where y is height
    Output: normalized tensor of same shape where peak y = 1 and center at (0, 0) in x,z
    """
    min_coords = dsm_points.min(dim=1, keepdim=True)[0]      # (B, 1, 3)
    max_coords = dsm_points.max(dim=1, keepdim=True)[0]      # (B, 1, 3)
    
    # Use y-range (height range) as the scaling factor for isotropic scaling
    y_range = max_coords[:, :, 1:2] - min_coords[:, :, 1:2] + 1e-8  # (B, 1, 1)
    
    # Scale all dimensions by the y-range so that max y becomes 1
    normalized = (dsm_points - min_coords) / y_range         # isotropic scaling based on height
    
    # Center the x and z coordinates at 0
    center_xz = normalized.mean(dim=1, keepdim=True)         # (B, 1, 3) - get centers
    center_xz[:, :, 1] = 0                                   # Don't center y-axis (keep ground at 0)
    normalized = normalized - center_xz                      # Center x and z at 0
    
    return normalized

def denormalize_from_unit_cube(normalized_points, center, scale):
    """
    Reverts normalized points in [0, 1]^3 back to original coordinates.
    Args:
        normalized_points: list of [Ni, 3] tensors (variable sizes)
        center: Tensor of shape [B, 3]
        scale: Tensor of shape [B]
    Returns:
        list of denormalized [Ni, 3] tensors
    """
    denormalized = []

    for i, points in enumerate(normalized_points):
        device = points.device
        c = center[i].to(device)  # (3,)
        s = scale[i].to(device)   # scalar or shape (1,)

        pts = (points - 0.5) * s + c  # (Ni, 3)
        denormalized.append(pts)

    return denormalized
    
def compute_tree_volume(points):
    """
    Compute tree bounding box volume for determining number of points.
    
    Args:
        points: Tensor of shape (N, 3) representing point coordinates
        
    Returns:
        volume: Scalar tensor representing the bounding box volume
    """
    if points.numel() == 0:
        return torch.tensor(1e-6, device=points.device)
    
    min_bounds = points.min(dim=0)[0]  # (3,)
    max_bounds = points.max(dim=0)[0]  # (3,)
    volume = torch.prod(max_bounds - min_bounds)
    return torch.clamp(volume, min=1e-6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsm_dir', type=str, default='./validation/landmarks_austria/DSM/')
    parser.add_argument('--ortho_dir', type=str, default='./validation/landmarks_austria/ORTHOPHOTOS/')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--num_query_points', type=int, default=10000)
    parser.add_argument('--top_k', type=int, default=2500)
    parser.add_argument('--num_points', type=int, default=2500)
    parser.add_argument('--num_trees', type=int, default=100)
    parser.add_argument('--deciduous', type=bool, default=False) 
    parser.add_argument('--variable', type=str, default='1')
    parser.add_argument('--model', type=str, default='1') # 1: tree, 2: lf
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--top_k_max', type=int, default=10000)  # for variable top_k
    parser.add_argument('--dsm_convex_hull', type=bool, default=False) 
    parser.add_argument('--no_norm_orthophoto', type=bool, default=False)  # If True, do not normalize orthophoto images
    args = parser.parse_args()

    vis = visdom.Visdom(port=8099, env='val_' + args.env)
    vis.close(win=None)

    if args.model == '1': # normal
        from model_normal import TreeReconstructionNet
    elif args.model == '2': # DSM 
        from model_dsm import TreeReconstructionNet
    elif args.model == '3': # ortho
        from model_ortho import TreeReconstructionNet
    elif args.model == '4': # for variety 
        from model_normal_with_noise import TreeReconstructionNet
    elif args.model == '5': # ortho + dsm
        from model_colors import TreeReconstructionNet

    from scipy.spatial import ConvexHull

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TreeDataset(npoints=args.num_points, train=False, num_trees=args.num_trees, deciduous=args.deciduous, rootimg=args.ortho_dir, dsm_root=args.dsm_dir)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False, collate_fn=collate_fn)

    model = TreeReconstructionNet(num_points=args.num_points).to(device)
    checkpoint = torch.load(checkpoint_path := f"./log/{args.env}/network.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    output_dir = f"./landmarks_austria/TREE_MODELS/{args.env}"
    os.makedirs(output_dir, exist_ok=True)

    base_top_k = args.top_k        
    reference_scale = 1.0         
    print("Scanning all scale_dsm to determine max...")
    all_scales = []
    for _, _, _, _, _, scale_dsm_batch, _ in dataloader:
        all_scales.append(scale_dsm_batch)  # list of [B] tensors
    all_scales = torch.cat(all_scales)  # [total_trees]
    global_max_scale = all_scales.max()
    print(f"Max scale_dsm across dataset: {global_max_scale.item():.3f}")

    loop = 0
    with torch.no_grad():
        for ortho_img, dsm_points, index_view, filename, center_dsm, scale_dsm, original_dsm in dataloader:
            print('filename:', filename)

            # only check the filename with '13'
            # if not any('13' in f for f in filename) and not any('72' in f for f in filename) and not any('21' in f for f in filename):
            #     continue
            
            ortho_img, dsm_points = ortho_img.to(device), dsm_points.to(device)
            original_dsm = original_dsm.to(device)
            B = dsm_points.shape[0]

            # dsm_points = normalize_dsm_to_unit_top(dsm_points)

            # ortho_img = (ortho_img - 0.5) / 0.5
            if args.no_norm_orthophoto == False:
                ortho_img = (ortho_img - 0.5) / 0.5
            if args.dsm_convex_hull:
                # Sample points in the convex hull of the DSM points
                dsm_points = sample_points_in_convex_hull(B, dsm_points, num_query_points=args.num_points*2)

            resolution=args.resolution

            # Query points generation
            if args.variable == '1':
                query_points = create_query_points(B, dsm_points, num_query_points=args.num_query).to(device)
            elif args.variable == '2':
                query_points = sample_points_in_convex_hull(B, dsm_points, num_query_points=args.num_query)
            elif args.variable == '3':
                query_points = create_normalized_query_points(B, num_query_points=args.num_query)

            # vis.scatter(
            #     X=query_points[0][:, [0, 2, 1]],
            #     win='query',
            #     opts=dict(
            #         title="QUERY Points",
            #         markersize=2
            #     )
            # )
            
            # # HEIGHT-BASED SCALING
            # min_top_k = args.top_k
            # max_top_k = args.top_k_max
            # rel_scales = scale_dsm / global_max_scale
            # rel_scales = (rel_scales - rel_scales.min()) / (rel_scales.max() - rel_scales.min() + 1e-8)
            # top_k = (min_top_k + rel_scales * (max_top_k - min_top_k)).long()

            
            # VOLUME-BASED SCALING (Convex Hull Volume)
            min_top_k = args.top_k
            max_top_k = args.top_k_max

            # Compute convex hull volume for each DSM point cloud
            volumes = []
            for b in range(B):
                dsm_np = dsm_points[b].cpu().numpy()
                if dsm_np.shape[0] >= 4:
                    try:
                        hull = ConvexHull(dsm_np)
                        volume = hull.volume
                    except Exception:
                        volume = 1e-6
                else:
                    volume = 1e-6
                volumes.append(torch.tensor(volume, device=dsm_points.device))
            volumes = torch.stack(volumes)

            # Normalize volumes relative to max volume in dataset for consistent scaling
            global_max_volume = volumes.max()
            if global_max_volume > 0:
                rel_volumes = volumes / global_max_volume
                rel_volumes = (rel_volumes - rel_volumes.min()) / (rel_volumes.max() - rel_volumes.min() + 1e-8)
                top_k = (min_top_k + rel_volumes * (max_top_k - min_top_k)).long()
            else:
                top_k = torch.full((B,), min_top_k, dtype=torch.long)


            if args.model in ['1', '2', '3', '4']:
                occupancy_pred, class_logits = model(dsm_points, ortho_img, query_points)
            elif args.model == '5':
                occupancy_pred, class_logits, _ = model(dsm_points, ortho_img, query_points)
            else:
                occupancy_pred = model(dsm_points, ortho_img, query_points)

            probs = torch.sigmoid(occupancy_pred)
            pred_points, top_k_indices = extract_top_k_occupied_points_val(probs, query_points, top_k)

            pred_points = denormalize_from_unit_cube(pred_points, center_dsm, scale_dsm)
            dsm_points = denormalize_from_unit_cube(dsm_points, center_dsm, scale_dsm) # _DSM

            for p in range(B):
                # if loop > 0:
                #     continue
                if p > 5: 
                    break

                # check if filename contains '4' or '5'
                # Only process files whose name is exactly '4', '5', '6', or '3' (not e.g. '35')
                base = os.path.splitext(os.path.basename(filename[p]))[0]
                tree_id = base.split('_')[1]  # e.g. 'tree_4' -> '4'
                # if tree_id not in {'1', '2', '3', '4', '5', '6'}:
                #     continue

                all_points = torch.cat([
                    pred_points[p][:, [0, 2, 1]],                  
                    #query_points[p][:, [0, 2, 1]],
                    dsm_points[p][:, [0, 2, 1]]               
                ], dim=0)
                labels = torch.cat([
                    torch.ones(pred_points[p].shape[0]),                 
                    #2 * torch.ones(query_points[0].shape[0]),
                    2 * torch.ones(dsm_points[p].shape[0])
                ])
                vis.scatter(
                    X=all_points,
                    Y=labels,
                    opts=dict(
                        title=f"Results - {filename[p]}",
                        markersize=2,
                        legend=["Predicted", "DSM"], # , "Query"
                    )
                )
                vis.image(
                    (ortho_img[p] * 255).byte().cpu().numpy(),
                    opts=dict(title=f"Ortho Image - {filename[p]}"),
                )

                # vis.scatter(
                #     X=original_dsm[p][:, [0, 2, 1]],
                #     opts=dict(
                #         title=f"Original DSM - {p}",
                #         markersize=2,
                #         legend=["Original DSM"]
                #     )
                # )

            # Only save files that contain '13'
            for i in range(0, len(pred_points)):
                # if '13' in filename[i] or '72' in filename[i]  or '21' in filename[i]: ## 13 filename
                ply_folder = os.path.join(output_dir, "pointclouds-landmarks")
                os.makedirs(ply_folder, exist_ok=True)
                ply_filename = os.path.join(ply_folder, f"{os.path.basename(filename[i])}.ply")
                save_points_as_ply(pred_points[i], ply_filename)

                print(f"Generated {top_k[i]} occupied points for {filename[i]}, scale: {scale_dsm[i].item()}")

            loop += 1

            # # Reconstruction mesh generation (validation/various/reconstruction.py)
            # for b in range(B):
            #     if '13' in filename[b]:
            #         mesh = alpha_shape_mesh(
            #             pred_points[b] # , radius=radius
            #         )

            #         filename_base = os.path.basename(filename[b])
            #         output_path = os.path.join(output_dir, f"{filename_base}_mesh.obj")
            #         o3d.io.write_triangle_mesh(output_path, mesh)

# CUDA_VISIBLE_DEVICES=2 python validation_pipeline_landmarks.py --env p2-occ_con_dec_shadow_silh --num_query_points 12000 --top_k 2500 --num_points 2500

# rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/P2-OCC/landmarks_austria/ "/mnt/c/Users/mmddd/Documents/P2-OCC/landmarks_austria"

# CUDA_VISIBLE_DEVICES=1 python validation/validation_pipeline_landmarks.py --env occ_2 --num_query_points 20000 --top_k 2500 --num_points 5000 --variable 1