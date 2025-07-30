import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse
import numpy as np
import trimesh
from auxiliary.occupancy_compute import compute_occupancy_top_k
import visdom
from sklearn.neighbors import NearestNeighbors

sys.path.append('./auxiliary/')
from query_points_gen import *
from occupancy_compute import extract_top_k_occupied_points, compute_occupancy, extract_threshold_occupied_points
from loss_functions import reconstruction_loss

sys.path.append('./auxiliary/dataset')
from dataset_test import TreeDataset

def save_points_as_ply(points, filename):
    """Saves points as a PLY file."""
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    cloud = trimesh.PointCloud(points)
    cloud.export(filename)

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    imgs, points, initial_vertices, index_views, filenames, center_dsm, scale_dsm = zip(*batch)

    imgs = torch.stack([torch.tensor(img, dtype=torch.float32) if isinstance(img, np.ndarray) else img for img in imgs])
    points = torch.stack([torch.tensor(p, dtype=torch.float32) if isinstance(p, np.ndarray) else p for p in points])
    initial_vertices = torch.stack([torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v for v in initial_vertices])
    index_views = torch.tensor(index_views)
    
    center_dsm = torch.stack([
        torch.tensor(c, dtype=torch.float32) if not torch.is_tensor(c) else c
        for c in center_dsm
    ])
    scale_dsm = torch.stack([
        torch.tensor([s], dtype=torch.float32) if not torch.is_tensor(s) else s
        for s in scale_dsm
    ]).squeeze(1)

    return imgs, points, initial_vertices, index_views, list(filenames), center_dsm, scale_dsm

def compute_chamfer(pred_points, gt_points):
    """Computes Chamfer RMSE (in meters) between predicted and ground truth point clouds."""
    cd_squared = reconstruction_loss(pred_points, gt_points)
    cd_rmse = torch.sqrt(cd_squared)
    return cd_rmse

def pad_or_trim_tensor(tensor, target_num_points):
    """Pad or trim tensor to target number of points."""
    N = tensor.shape[0]
    if N == 0:
        return torch.zeros((target_num_points, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    if N == target_num_points:
        return tensor
    elif N > target_num_points:
        return tensor[:target_num_points]
    else:  # Pad
        pad_size = target_num_points - N
        pad = tensor[torch.randint(0, N, (pad_size,), device=tensor.device)]
        return torch.cat([tensor, pad], dim=0)

def compute_tree_height(points):
    """Compute tree height assuming Y-axis is vertical."""
    return points[:, 1].max() - points[:, 1].min()

def compute_mean_chamfer(chamfers):
    """Compute mean RMSE from squared chamfer distances."""
    chamfers_tensor = torch.tensor(chamfers)
    rmse_per_pair = torch.sqrt(chamfers_tensor)
    return rmse_per_pair.mean().item()

def compute_f1_score(pred_points, gt_points, threshold=0.01):
    """
    Compute F1 score between predicted and ground truth point clouds.
    
    Args:
        pred_points: Predicted point cloud (B, N, 3)
        gt_points: Ground truth point cloud (B, M, 3)
        threshold: Distance threshold for considering a match
    
    Returns:
        f1_score: F1 score
        precision: Precision score
        recall: Recall score
    """
    from sklearn.neighbors import NearestNeighbors
    
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.cpu().numpy()
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.cpu().numpy()
    
    # Flatten if batched
    if pred_points.ndim == 3:
        pred_points = pred_points.reshape(-1, 3)
    if gt_points.ndim == 3:
        gt_points = gt_points.reshape(-1, 3)
    
    # Build KNN for GT points
    nbrs_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(gt_points)
    distances_pred_to_gt, _ = nbrs_gt.kneighbors(pred_points)
    
    # Build KNN for predicted points
    nbrs_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pred_points)
    distances_gt_to_pred, _ = nbrs_pred.kneighbors(gt_points)
    
    # Precision: fraction of predicted points within threshold of GT
    precision = np.mean(distances_pred_to_gt.flatten() < threshold)
    
    # Recall: fraction of GT points within threshold of predicted
    recall = np.mean(distances_gt_to_pred.flatten() < threshold)
    
    # F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score, precision, recall

def compute_coverage_mmd_cov(all_pred_points, all_gt_points, k=5):
    """
    Compute Coverage, Maximum Mean Discrepancy (MMD), and COV metrics.
    
    Args:
        all_pred_points: List of predicted point clouds
        all_gt_points: List of ground truth point clouds
        k: Number of nearest neighbors for coverage computation
    
    Returns:
        coverage: Coverage score (higher is better)
        mmd: Maximum Mean Discrepancy (lower is better)
        cov: COV score (higher is better, measures diversity)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Convert to numpy arrays
    pred_features = []
    gt_features = []
    
    for pred, gt in zip(all_pred_points, all_gt_points):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # Use mean coordinates as features (simple feature extraction)
        pred_feat = np.array([pred.mean(axis=0).flatten()])  # Shape: (1, 3)
        gt_feat = np.array([gt.mean(axis=0).flatten()])      # Shape: (1, 3)
        
        pred_features.append(pred_feat.flatten())
        gt_features.append(gt_feat.flatten())
    
    pred_features = np.array(pred_features)  # Shape: (N, 3)
    gt_features = np.array(gt_features)      # Shape: (N, 3)
    
    # Coverage: For each GT sample, check if there's a predicted sample within k-NN
    nbrs_pred = NearestNeighbors(n_neighbors=min(k, len(pred_features)), algorithm='kd_tree').fit(pred_features)
    distances_gt_to_pred, _ = nbrs_pred.kneighbors(gt_features)
    
    # Coverage is the fraction of GT samples that have a close predicted sample
    coverage_threshold = np.percentile(distances_gt_to_pred.min(axis=1), 50)  # Median distance as threshold
    coverage = np.mean(distances_gt_to_pred.min(axis=1) < coverage_threshold)
    
    # MMD: Average minimum distance from predicted to GT
    nbrs_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(gt_features)
    distances_pred_to_gt, _ = nbrs_gt.kneighbors(pred_features)
    mmd = np.mean(distances_pred_to_gt.flatten())
    
    # COV: Diversity measure - average pairwise distance in predicted set
    if len(pred_features) > 1:
        pairwise_distances = []
        for i in range(len(pred_features)):
            for j in range(i+1, len(pred_features)):
                dist = np.linalg.norm(pred_features[i] - pred_features[j])
                pairwise_distances.append(dist)
        cov = np.mean(pairwise_distances) if pairwise_distances else 0.0
    else:
        cov = 0.0
    
    return coverage, mmd, cov

def print_evaluation_results(results):
    """Print concise evaluation results for reporting."""
    print("\n" + "="*80)
    print("TREE RECONSTRUCTION EVALUATION RESULTS 🌲")
    print("="*80)
    
    # Geometric Accuracy
    print(f"\nGEOMETRIC ACCURACY:")
    print(f"{'Chamfer Distance (RMSE):':<30} {results['chamfer_rmse']:.4f} meters")
    print(f"{'Normalized Chamfer Distance:':<30} {results['normalized_chamfer']:.4f}")
    
    # Completeness & Precision
    print(f"\nCOMPLETENESS & PRECISION:")
    print(f"{'F1 Score:':<30} {results['f1_score']:.4f}")
    
    # Diversity
    print(f"\nDIVERSITY:")
    print(f"{'Variance Score (COV):':<30} {results['variance_score']:.2f}%")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

def denormalize_from_unit_cube(normalized_points, center, scale):
    """Reverts normalized points in [0, 1]^3 back to original coordinates."""
    denormalized = []
    for i, points in enumerate(normalized_points):
        device = points.device
        c = center[i].to(device)
        s = scale[i].to(device)
        pts = (points - 0.5) * s + c
        denormalized.append(pts)
    return denormalized

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
    parser.add_argument('--variable', type=str, default='0')
    parser.add_argument('--num_trees_total', type=int, default=100)
    parser.add_argument('--model', type=str, default='1')
    parser.add_argument('--dsm_convex_hull', type=bool, default=False)
    parser.add_argument('--no_norm_orthophoto', type=bool, default=False)
    parser.add_argument('--top_k_max', type=int, default=2500)
    parser.add_argument('--top_k_gt_occupancy', type=bool, default=False, help='Use top-k occupancy for GT points')
    args = parser.parse_args()

    # Initialize Visdom
    vis = visdom.Visdom(port=8099, env='val_' + args.env)
    vis.close(win=None)

    # Import model based on model type
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup dataset
    filename = f"val_npy/fixed_trees.npy"
    
    # Check if we have fixed tree IDs saved from previous runs
    if os.path.exists(filename):
        print(f"Loading fixed tree IDs from {filename}")
        ids = np.load(filename, allow_pickle=True)
        print(f"Loaded {len(ids)} fixed tree IDs")
        
        # Create dataset with fixed IDs (100% test data)
        try:
            dataset = TreeDataset(
                npoints=args.num_points, 
                train=False, 
                num_trees=args.num_trees_total, 
                deciduous=args.deciduous, 
                fixed_ids=ids
            )
            print(f"Successfully created dataset with fixed IDs: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Error creating dataset with fixed_ids: {e}")
            print("Falling back to creating new dataset...")
            # Fallback to creating new dataset
            dataset = TreeDataset(
                npoints=args.num_points, 
                train=False, 
                num_trees=args.num_trees_total, 
                deciduous=args.deciduous
            )
            # Save the new tree IDs
            selected_tree_ids = dataset.get_filenames()
            np.save(filename, selected_tree_ids)
            print(f"Saved {len(selected_tree_ids)} new tree IDs to {filename}")
    else:
        print(f"No fixed tree IDs found. Creating new dataset...")
        # Create initial dataset to get filenames
        dataset = TreeDataset(
            npoints=args.num_points, 
            train=False, 
            num_trees=args.num_trees_total, 
            deciduous=args.deciduous
        )
        
        # Save the selected tree IDs for consistent evaluation
        selected_tree_ids = dataset.get_filenames()
        os.makedirs("val_npy", exist_ok=True)
        np.save(filename, selected_tree_ids)
        print(f"Saved {len(selected_tree_ids)} tree IDs to {filename} for future consistent evaluation")
    
    print(f"Final dataset size: {len(dataset)} samples")
    print("Using 100% of selected trees as test data (no train/val split in validation)")
    
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False, collate_fn=collate_fn)
    
    # Load model
    model = TreeReconstructionNet(num_points=args.num_points).to(device)
    checkpoint = torch.load(f"./log/{args.env}/network.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        all_gt_points = []
        all_pred_points = []
        all_dsm_points = []
        all_filenames = []
        closest_matches = []

        all_gt_meshes = []
        
        for batch_idx, (orthophoto, gt_mesh, dsm_points, index_views, filenames, center_dsm, scale_dsm) in enumerate(dataloader):
            orthophoto, gt_mesh, dsm_points = orthophoto.to(device), gt_mesh.to(device), dsm_points.to(device)
            B = dsm_points.shape[0]

            # Normalize orthophoto
            if args.no_norm_orthophoto == False:
                orthophoto = (orthophoto - 0.5) / 0.5

            # Handle DSM convex hull sampling if needed
            if args.dsm_convex_hull:
                dsm_points = sample_points_in_convex_hull(B, dsm_points, num_query_points=args.num_points*2)

            # Query points generation
            if args.variable == '1':
                query_points = create_query_points(B, dsm_points, num_query_points=args.num_query).to(device)
            elif args.variable == '2':
                query_points = sample_points_in_convex_hull(B, dsm_points, num_query_points=args.num_query)
            elif args.variable == '3':
                query_points = create_normalized_query_points(B, num_query_points=args.num_query)

            # Compute ground truth occupancy
            if args.top_k_gt_occupancy:
                gt_occupancy = compute_occupancy_top_k(B, query_points, gt_mesh, top_k=opt.top_k)
            else: 
                gt_occupancy = compute_occupancy(B, query_points, gt_mesh, threshold=opt.thres) ## TOSEE
            gt_points = extract_threshold_occupied_points(gt_occupancy, query_points, threshold=0.5)

            # Model prediction
            if args.model in ['1', '2', '3', '4']:
                occupancy_pred, class_logits = model(dsm_points, orthophoto, query_points)
            elif args.model == '5':
                occupancy_pred, class_logits, _ = model(dsm_points, orthophoto, query_points)
            else: 
                occupancy_pred = model(dsm_points, orthophoto, query_points)
            
            probs = torch.sigmoid(occupancy_pred)
            pred_points, topk_indices = extract_top_k_occupied_points(probs, query_points, args.top_k)

            # Denormalize points back to original coordinates
            pred_points = denormalize_from_unit_cube(pred_points, center_dsm, scale_dsm)
            dsm_points = denormalize_from_unit_cube(dsm_points, center_dsm, scale_dsm)
            gt_points = denormalize_from_unit_cube(gt_points, center_dsm, scale_dsm)

            gt_mesh = denormalize_from_unit_cube(gt_mesh, center_dsm, scale_dsm)

            # Process each batch element
            for b in range(B):                
                pred_fixed = pad_or_trim_tensor(pred_points[b], args.num_points)
                gt_fixed = pad_or_trim_tensor(gt_points[b], args.num_points)

                gt_mesh_fixed = pad_or_trim_tensor(gt_mesh[b], args.num_points) 

                all_gt_points.append(gt_fixed)
                all_pred_points.append(pred_fixed)
                all_dsm_points.append(dsm_points[b])
                all_filenames.append(filenames[b])

                all_gt_meshes.append(gt_mesh_fixed)

        print(f'Total samples processed: {len(all_gt_points)}')
        for b, pred in enumerate(all_pred_points):
            pred = pred.unsqueeze(0) if pred.dim() == 2 else pred  # Ensure shape (B, N, 3)
            chamfers = []
            for i in range(len(all_gt_points)): 
                gt_candidate = all_gt_points[i].unsqueeze(0).to(device) if all_gt_points[i].dim() == 2 else all_gt_points[i].to(device)
                chamfer = compute_chamfer(pred, gt_candidate)
                chamfers.append(chamfer.cpu().item())  # Ensure chamfer is converted to a CPU scalar

            closest_idx = np.argmin(chamfers)
            closest_matches.append(closest_idx)

        # Count how many *unique* GT meshes were chosen (for comparison)
        unique_matches = len(set(closest_matches))
        total_predictions = len(closest_matches)

        print(f"\n📊 Inter-sample Statistics (for comparison):")
        print(f"Total Predictions: {total_predictions}")
        print(f"Unique GT Matches: {unique_matches}")

        variance_score = unique_matches / total_predictions  # Fraction of unique matches
        print(f"Inter-sample Variance: {variance_score * 100:.2f}% unique matches")

        for p in range(len(all_pred_points)):
            if p > 15:
                break
            all_points = torch.cat([
                all_pred_points[p][:, [0, 2, 1]],
                all_gt_points[p][:, [0, 2, 1]],
            ], dim=0)
            labels = torch.cat([
                torch.ones(all_gt_points[p].shape[0]),                 
                #2 * torch.ones(query_points[0].shape[0]),
                2 * torch.ones(all_gt_points[p].shape[0])
            ])
            vis.scatter(
                X=all_points,
                Y=labels,
                opts=dict(
                    title=f"Results - {p}",
                    markersize=2,
                    legend=["Predicted", "GT"], # , "Query"
                )
            )

        chamfer_distances = []
        normalized_chamfers = []
        all_pred_tensors = []
        all_gt_tensors = []
        
        for b, pred in enumerate(all_pred_points):
            pred = pred.unsqueeze(0) if pred.dim() == 2 else pred  # Ensure shape (B, N, 3)
            gt_candidate = all_gt_points[b].unsqueeze(0).to(device) if all_gt_points[b].dim() == 2 else all_gt_points[b].to(device)
            chamfer = compute_chamfer(pred, gt_candidate)
            tree_size = compute_tree_height(gt_candidate.squeeze(0))
            normalized_chamfer = chamfer / tree_size
            chamfer_distances.append(chamfer.cpu().item())  # Ensure chamfer is converted to a CPU scalar
            normalized_chamfers.append(normalized_chamfer.cpu().item())  # Ensure chamfer is converted to a CPU scalar
            
            # Store for additional metrics
            all_pred_tensors.append(pred.squeeze(0).cpu())
            all_gt_tensors.append(gt_candidate.squeeze(0).cpu())

        # Compute comprehensive metrics
        avg_chamfer_rmse = compute_mean_chamfer(chamfer_distances)
        chamfer_std = np.std(chamfer_distances)
        avg_chamfer_norm = compute_mean_chamfer(normalized_chamfers)
        
        # Compute F1 score
        all_pred_combined = torch.cat(all_pred_tensors, dim=0)
        all_gt_combined = torch.cat(all_gt_tensors, dim=0)
        f1_score, precision, recall = compute_f1_score(all_pred_combined, all_gt_combined, threshold=0.02)
        
        # Compute Coverage, MMD, COV metrics
        coverage, mmd, cov = compute_coverage_mmd_cov(all_pred_tensors, all_gt_tensors)
        
        # Calculate variance in predictions (fraction of unique matches)
        variance_score = unique_matches / total_predictions * 100  # Percentage of unique matches
        
        # Organize results
        results = {
            'total_samples': len(all_pred_points),
            'num_query_points': args.num_query_points,
            'num_points': args.num_points,
            'chamfer_rmse': avg_chamfer_rmse,
            'chamfer_std': chamfer_std,
            'normalized_chamfer': avg_chamfer_norm,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'coverage': coverage,
            'mmd': mmd,
            'cov': cov,
            'unique_matches': unique_matches,
            'total_predictions': total_predictions,
            'variance_score': variance_score
        }
        
        # Print organized results
        print_evaluation_results(results)

        # Create export folder
        # export_dir = f"./landmarks_austria/{args.env}/pointclouds-landmarks" # pointclouds"
        output_dir = f"./landmarks_austria/TREE_MODELS/{args.env}/"
        export_dir = os.path.join(output_dir, "pointclouds-validation")
        os.makedirs(export_dir, exist_ok=True)
        for p in range(len(all_pred_points)):
            pred_pts = all_pred_points[p][:, [0, 2, 1]]  # flip Y-Z
            gt_pts = all_gt_points[p][:, [0, 2, 1]]
            dsm_pts = all_dsm_points[p][:, [0, 2, 1]]  # assuming query ≈ DSM
            gt_mesh_pts = all_gt_meshes[p][:, [0, 2, 1]]  # flip Y-Z for GT mesh
            filename = all_filenames[p]

            save_points_as_ply(pred_pts, filename=f"{export_dir}/tree_{p:03d}_pred.ply")
            save_points_as_ply(gt_pts, filename=f"{export_dir}/tree_{p:03d}_gt.ply")
            save_points_as_ply(dsm_pts, filename=f"{export_dir}/tree_{p:03d}_dsm.ply")
            save_points_as_ply(gt_mesh_pts, filename=f"{export_dir}/tree_{p:03d}_gt_mesh.ply")

            # vis.scatter(
            #     X=gt_pts,
            #     Y=torch.ones(gt_pts.shape[0]),
            #     opts=dict(
            #         title=f"Predicted - {p} - {filename}",
            #         markersize=2,
            #         legend=["Predicted"],
            #     )
            # )


           # print(f"Saved tree {p:03d} PLY files.")

# CUDA_VISIBLE_DEVICES=2 python validation_pipeline_landmarks.py --env p2-occ_con_dec_shadow_silh --num_query_points 12000 --top_k 2500 --num_points 2500

# rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/P2-OCC/landmarks_austria/ "/mnt/c/Users/mmddd/Documents/P2-OCC/landmarks_austria"

# CUDA_VISIBLE_DEVICES=1 python validation/validation_pipeline_landmarks.py --env occ_2 --num_query_points 20000 --top_k 2500 --num_points 5000 --variable 1

# CUDA_VISIBLE_DEVICES=1 python validation/validation_pipeline.py --env test3 --num_query_points 20000 --top_k 5000 --num_points 5000 --model 4 --deciduous true --num_trees_total 35 --variable 1
##  rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/p2-tree-gen/landmarks_austria/test3/export-ply "/mnt/c/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/models"


# sed -i 's/\r$//' run_all_validations.sh
# ./run_all_validations.sh