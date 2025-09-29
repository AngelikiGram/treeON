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

sys.path.append('./auxiliary/models')
def save_points_as_ply(points, filename, colors=None):
    """
    Saves the given points (tensor or list of tensors) as a PLY file.
    points: (N, 3) tensor/ndarray or list of (Ni, 3) tensors/ndarrays.
    colors: (N, 3) tensor/ndarray or list of (Ni, 3) tensors/ndarrays, values in [0, 255] or [0, 1].
    filename: output path (.ply).
    """
    # Handle list of tensors/arrays
    if isinstance(points, (list, tuple)):
        pts_list = []
        for p in points:
            if isinstance(p, torch.Tensor):
                pts_list.append(p.cpu().numpy())
            else:
                pts_list.append(np.asarray(p))
        points = np.concatenate(pts_list, axis=0)
    elif isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    else:
        points = np.asarray(points)

    N = points.shape[0]

    # Handle colors
    if colors is not None:
        if isinstance(colors, (list, tuple)):
            col_list = []
            for c in colors:
                if isinstance(c, torch.Tensor):
                    col_list.append(c.cpu().numpy())
                else:
                    col_list.append(np.asarray(c))
            colors = np.concatenate(col_list, axis=0)
        elif isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        else:
            colors = np.asarray(colors)
        if colors.shape[1] > 3:
            colors = colors[:, :3]
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        # Ensure colors array matches number of points
        if colors.shape[0] < N:
            # Pad colors with zeros if too short
            pad = np.zeros((N - colors.shape[0], 3), dtype=np.uint8)
            colors = np.concatenate([colors, pad], axis=0)
        elif colors.shape[0] > N:
            # Truncate colors if too long
            colors = colors[:N]
        # Write manual ASCII PLY with colors
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % N)
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(N):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f'{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n')
    else:
        # Write manual ASCII PLY without colors
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % N)
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(N):
                x, y, z = points[i]
                f.write(f'{x:.8f} {y:.8f} {z:.8f}\n')

def collate_fn(batch):
    imgs, initial_vertices, index_views, filenames, center_dsm, scale_dsm, original_dsm, labels = zip(*batch)

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
    labels = torch.tensor(labels)

    filename = [f for f in filenames]

    center_dsm = torch.stack([
        torch.tensor(c, dtype=torch.float32) if not torch.is_tensor(c) else c
        for c in center_dsm
    ])

    scale_dsm = torch.stack([
        torch.tensor([s], dtype=torch.float32) if not torch.is_tensor(s) else s
        for s in scale_dsm
    ]).squeeze(1)  # Squeeze to make shape [B] instead of [B, 1]

    return imgs, initial_vertices, index_views, filename, center_dsm, scale_dsm, original_dsm, labels

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

import torch
import torch.nn.functional as F
import numpy as np

def histogram_match_gt_colors(
    gt_mesh,
    gt_colors,
    orthophoto,
    *,
    center_dsm=None,          # [B,3] or None. If provided, we invert (x) = (x - c)/s + 0.5
    scale_dsm=None,           # [B] or [B,1] or None
    assume_normalized=False,  # True if gt_mesh x,z already in [0,1]
    blend_ref=0.3,            # Final = (1-blend_ref)*sampled + blend_ref*gt_colors
    bins=256,
    clamp_coords=True,
    align_corners=True
):
    """
    Histogram-match each orthophoto to the reference color distribution (gt_colors),
    then sample colors at (x,z) from the matched orthophoto.

    Args:
        gt_mesh:   Tensor [B,P,3] or list of length B with tensors [Pi,3].
        gt_colors: Tensor [B,P,3] or list of length B with tensors [Pi,3] or tensor [B,*,3].
                   Used only as the *reference* distribution for histogram matching (P may differ).
        orthophoto: Tensor [B,3,H,W].
        center_dsm, scale_dsm: Optional tensors for de-normalization inverse. If provided,
                   coords are mapped via (x - c)/s + 0.5 before grid_sample.
        assume_normalized: If True, gt_mesh x,z are already in [0,1].
        blend_ref: Blend with input gt_colors (0 => use sampled only).
    Returns:
        matched_sampled_colors:
            - If gt_mesh is a list: returns list of [Pi,3] tensors.
            - If gt_mesh is a tensor: returns tensor [B,P,3].
    """
    # Helpers to unify list/tensor handling
    def _is_list(x): return isinstance(x, (list, tuple))
    def _as_list(x):
        if _is_list(x): return list(x)
        return [x[b] for b in range(x.shape[0])]
    def _stack_like(ref, lst):
        if _is_list(ref):
            return lst
        return torch.nn.utils.rnn.pad_sequence(lst, batch_first=True) if any(
            t.shape[0] != lst[0].shape[0] for t in lst
        ) else torch.stack(lst, dim=0)

    device = orthophoto.device
    dtype  = orthophoto.dtype

    B = orthophoto.shape[0]
    meshes  = _as_list(gt_mesh)     # list of [Pi,3]
    colors  = _as_list(gt_colors)   # list of [Qi,3] (only distribution is used)
    centers = None if center_dsm is None else (center_dsm if _is_list(center_dsm) else [center_dsm[b] for b in range(B)])
    scales  = None if scale_dsm  is None else (scale_dsm  if _is_list(scale_dsm)  else [scale_dsm[b]  for b in range(B)])

    # Normalize orthophoto to [0,1] if needed
    with torch.no_grad():
        o = orthophoto
        if o.min() < -0.1:   # heuristic for [-1,1]
            o = (o + 1.0) * 0.5
        o = o.clamp(0.0, 1.0)

    # Try skimage; otherwise use a lightweight CDF matcher
    try:
        from skimage.exposure import match_histograms as _sk_match
        def _match_rgb(src_hw3, ref_p3):
            # Ensure ref_p3 is 3D for skimage
            if ref_p3.ndim == 2:
                # If shape is (N, 3), tile or reshape to match src_hw3's shape
                H, W, C = src_hw3.shape
                if ref_p3.shape[0] == H * W:
                    ref_p3 = ref_p3.reshape(H, W, C)
                else:
                    # fallback: tile mean color
                    mean_color = np.mean(ref_p3, axis=0)
                    ref_p3 = np.tile(mean_color, (H, W, 1))
            return _sk_match(src_hw3, ref_p3, channel_axis=2)
    except Exception:
        def _match_1d(src_flat, ref_flat, nbins=bins):
            src_flat = np.asarray(src_flat, dtype=np.float64)
            ref_flat = np.asarray(ref_flat, dtype=np.float64)
            hist_src, bin_edges = np.histogram(src_flat, bins=nbins, range=(0.0, 1.0), density=True)
            hist_ref, _        = np.histogram(ref_flat, bins=nbins, range=(0.0, 1.0), density=True)
            cdf_src = np.cumsum(hist_src); cdf_src /= (cdf_src[-1] + 1e-12)
            cdf_ref = np.cumsum(hist_ref); cdf_ref /= (cdf_ref[-1] + 1e-12)
            centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
            src_cdf_vals = np.interp(src_flat, centers, cdf_src)
            matched = np.interp(src_cdf_vals, cdf_ref, centers)
            return np.clip(matched, 0.0, 1.0)
        def _match_rgb(src_hw3, ref_p3):
            H_, W_, _ = src_hw3.shape
            src_flat = src_hw3.reshape(-1, 3)
            ref_flat = ref_p3.reshape(-1, 3)
            out = np.empty_like(src_flat)
            for c in range(3):
                out[:, c] = _match_1d(src_flat[:, c], ref_flat[:, c])
            return out.reshape(H_, W_, 3)

    outs = []
    for b in range(B):
        pts = meshes[b].to(device=device, dtype=dtype)           # [Pi,3]
        ref = colors[b].to(device=device, dtype=dtype)           # [Qi,3] (any Qi)
        H, W = o.shape[2], o.shape[3]

        # Build (x,z) -> [0,1] coords
        xz = pts[:, [0, 2]]

        if not assume_normalized:
            if (centers is not None) and (scales is not None):
                c = centers[b].to(device=device, dtype=dtype)    # [3]
                s = scales[b].to(device=device, dtype=dtype)     # [] or [1]
                if s.ndim > 0: s = s.squeeze()
                xz = ((pts - c) / s + 0.5)[:, [0, 2]]            # invert (x) = (x - c)/s + 0.5
            else:
                # Fallback: per-batch min-max (keeps everything in-bounds, but may be misaligned)
                mins = xz.min(dim=0, keepdim=True)[0]
                maxs = xz.max(dim=0, keepdim=True)[0]
                xz = (xz - mins) / (maxs - mins + 1e-8)

        if clamp_coords:
            xz = xz.clamp(0.0, 1.0)

        grid = (xz * 2.0 - 1.0).unsqueeze(1)  # [Pi,2] -> [Pi,1,2]

        # Prepare numpy arrays for matching
        ortho_b = o[b].detach().cpu().numpy().transpose(1, 2, 0)     # [H,W,3]
        ref_b   = ref.detach().cpu().numpy().reshape(-1, 3)          # [Qi,3]

        # Histogram-match orthophoto to ref distribution
        ortho_matched = _match_rgb(ortho_b, ref_b)                    # [H,W,3] in [0,1]

        # Back to tensor and sample
        ortho_matched_t = torch.from_numpy(ortho_matched).to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        sampled = F.grid_sample(
            ortho_matched_t, grid.unsqueeze(0), mode='bilinear',
            padding_mode='zeros', align_corners=align_corners
        )  # [1,3,Pi,1]
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)        # [Pi,3]

        if blend_ref > 0:
            # If ref has a different length than Pi, broadcast by nearest (simple, robust).
            if ref.shape[0] != pts.shape[0]:
                # Nearest index mapping
                idx = torch.linspace(0, ref.shape[0]-1, steps=pts.shape[0], device=device).round().long()
                ref_use = ref[idx]
            else:
                ref_use = ref
            # Blend 25% original (ref_use) and 75% ortho blend (sampled)
            out = 0.2 * sampled + 0.8 * ref_use
        else:
            out = sampled

        outs.append(out.clamp(0.0, 1.0))

    return _stack_like(gt_mesh, outs)

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
    parser.add_argument('--dsm_dir', type=str, default='./landmarks_austria/DATA_LANDMARKS/DSM/')
    parser.add_argument('--ortho_dir', type=str, default='./landmarks_austria/DATA_LANDMARKS/ORTHOPHOTOS/')
    parser.add_argument('--species_file', type=str, default=None, help='Path to species information file')
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
    parser.add_argument('--log_dir', type=str, default='log_p2', help='Log directory name (default: log_p2)')
    args = parser.parse_args()

    vis = visdom.Visdom(port=8099, env='val_' + args.env)
    vis.close(win=None)

    if args.model == '1': # normal
        from model_normal_categories import TreeReconstructionNet # categories
    elif args.model == '2': # DSM 
        from model_dsm import TreeReconstructionNet
    elif args.model == '3': # ortho
        from model_ortho import TreeReconstructionNet
    elif args.model == '4': # for variety 
        from model_normal_with_noise import TreeReconstructionNet    
    elif args.model == '10': # dsm with noise
        from model_normal_categories_noRefinement import TreeReconstructionNet
    elif args.model == '11': # ortho + dsm + noise
        from model_colors_old import TreeReconstructionNet
    elif args.model == '5': # ortho + dsm
        from model_colors import TreeReconstructionNet

    from scipy.spatial import ConvexHull

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TreeDataset(npoints=args.num_points, train=False, num_trees=args.num_trees, deciduous=args.deciduous, rootimg=args.ortho_dir, dsm_root=args.dsm_dir, species_file=args.species_file)
    
    print(f"📊 Dataset summary:")
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Root image dir: {args.ortho_dir}")
    print(f"   DSM root dir: {args.dsm_dir}")
    print(f"   Species file: {args.species_file}")
    
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False, collate_fn=collate_fn)

    # Print species information
    checkpoint = torch.load(checkpoint_path := f"./{args.log_dir}/{args.env}/network.pth")
    
    num_species = 14 # 14

    model = TreeReconstructionNet(num_points=args.num_points, num_species=num_species).to(device)
    
    # Check if the checkpoint matches the current model configuration
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        if "size mismatch for classifier" in str(e):
            print(f"❌ Model classifier size mismatch!")
            print(f"   Current model expects {num_species} species")
            print(f"   Checkpoint was trained with a different number of classes")
            print(f"   Please ensure you're using the same species file that was used during training")
            print(f"   Or retrain the model with the current species configuration")
            raise e
        else:
            raise e
    
    model.eval()

    output_dir = f"./landmarks_austria/TREE_MODELS/{args.env}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataloader has any data
    if len(dataloader) == 0:
        print("❌ No valid samples found in dataset!")
        print("   This could be due to:")
        print("   - No files matching the dataset criteria")
        print("   - Species filtering removed all files")
        print("   - Missing or corrupted data files")
        if args.species_file:
            print(f"   - Species file: {args.species_file}")
        print("   Please check your data paths and species file configuration.")
        exit(1)

    base_top_k = args.top_k        
    reference_scale = 1.0         
    print("Scanning all scale_dsm to determine max...")
    all_scales = []
    
    # Create a separate dataloader for scanning to avoid consuming the main one
    scan_dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False, collate_fn=collate_fn)
    for _, _, _, _, _, scale_dsm_batch, _, _ in scan_dataloader:
        all_scales.append(scale_dsm_batch)  # list of [B] tensors
    
    if len(all_scales) == 0:
        print("❌ No data batches found in dataloader!")
        exit(1)
        
    all_scales = torch.cat(all_scales)  # [total_trees]
    global_max_scale = all_scales.max()
    print(f"Max scale_dsm across dataset: {global_max_scale.item():.3f}")

    # Create fresh dataloader for main processing
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False, collate_fn=collate_fn)

    loop = 0
    
    all_pred_colors = []
    all_topk_indices = []
    with torch.no_grad():
        for ortho_img, dsm_points, index_view, filename, center_dsm, scale_dsm, original_dsm, labels in dataloader:
      
            # only check the filename with '13'
            # if not any('13' in f for f in filename) and not any('72' in f for f in filename) and not any('21' in f for f in filename):
            #     continue
            
            ortho_img, dsm_points = ortho_img.to(device), dsm_points.to(device)
            original_dsm = original_dsm.to(device)
            B = dsm_points.shape[0]

            # print the lowest dsm point in y axis
            print("Lowest DSM point in y-axis:", dsm_points[:, :, 1].min(dim=1).values)
            
            # Print DSM Z-axis min/max
            print("DSM Z-axis - Min:", dsm_points[:, :, 2].min(dim=1).values)
            print("DSM Z-axis - Max:", dsm_points[:, :, 2].max(dim=1).values)

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
                query_points = create_query_points(B, dsm_points, num_query_points=args.num_query_points).to(device)
            elif args.variable == '2':
                query_points = sample_points_in_convex_hull(B, dsm_points, num_query_points=args.num_query_points)
            elif args.variable == '3':
                query_points = create_normalized_query_points(B, num_query_points=args.num_query_points)

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

            if args.model in ['1', '4']:
                occupancy_pred, class_logits = model(dsm_points, ortho_img, query_points)
            elif args.model in ['5', '10', '11', '2', '3']:
                occupancy_pred, class_logits, pred_colors = model(dsm_points, ortho_img, query_points)
            else:
                occupancy_pred = model(dsm_points, ortho_img, query_points)

            probs = torch.sigmoid(occupancy_pred)
            pred_points, top_k_indices = extract_top_k_occupied_points_val(probs, query_points, top_k)

            # Print generated points Z-axis min/max
            pred_z_mins = torch.stack([p[:, 2].min() for p in pred_points])
            pred_z_maxs = torch.stack([p[:, 2].max() for p in pred_points])
            print("Generated points Z-axis - Min:", pred_z_mins)
            print("Generated points Z-axis - Max:", pred_z_maxs)

            # # print the lowest pred_points in y axis
            # print("Lowest pred_points in y-axis:", torch.stack([p[:, 1] for p in pred_points]).min(dim=1).values)

            pred_points = denormalize_from_unit_cube(pred_points, center_dsm, scale_dsm)
            dsm_points = denormalize_from_unit_cube(dsm_points, center_dsm, scale_dsm) # _DSM 

            print("denormalization applied.")

            if args.model == '5':
                pred_colors1 = histogram_match_gt_colors(pred_points, pred_colors, ortho_img)

            for p in range(B):
                # if loop > 0:
                #     continue
                # if p > 5: 
                #     break

                # check if filename contains '4' or '5'
                # Only process files whose name is exactly '4', '5', '6', or '3' (not e.g. '35')
                base = os.path.splitext(os.path.basename(filename[p]))[0]
                tree_id = base.split('_')[1]  # e.g. 'tree_4' -> '4'
                # if tree_id not in {'1', '2', '3', '4', '5', '6'}:
                #     continue

                # Show combined points (predicted + DSM) as before
                all_points = torch.cat([
                    pred_points[p][:, [0, 2, 1]],                  
                    dsm_points[p][:, [0, 2, 1]]               
                ], dim=0)
                labels = torch.cat([
                    torch.ones(pred_points[p].shape[0]),                 
                    2 * torch.ones(dsm_points[p].shape[0])
                ])
                vis.scatter(
                    X=all_points,
                    Y=labels,
                    opts=dict(
                        title=f"Results - {filename[p]}",
                        markersize=2,
                        legend=["Predicted", "DSM"],
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

            # Visualize predicted colors for top-k occupied points (model 5)
            if args.model == '5' or args.model == '11' or args.model == '10' or args.model == '2' or args.model == '3':
                for p in range(B):
                    # Get top-k indices for this batch item
                    probs_p = torch.sigmoid(occupancy_pred[p]).squeeze(-1)
                    # top_k_indices[p] is already computed by extract_top_k_occupied_points_val
                    topk_idx = top_k_indices[p]
                    pts_np = pred_points[p][:, [0, 2, 1]].cpu().numpy()  # [top_k, 3]
                    # Get predicted colors for top-k points
                    pred_colors_p = pred_colors[p]  # [N_query, 3]
                    if isinstance(pred_colors_p, torch.Tensor):
                        pred_colors_p = pred_colors_p.detach().cpu().numpy()
                    if pred_colors_p.shape[1] > 3:
                        pred_colors_p = pred_colors_p[:, :3]
                    if pred_colors_p.max() <= 1.0:
                        pred_colors_p = (pred_colors_p * 255).astype(np.uint8)
                    else:
                        pred_colors_p = pred_colors_p.astype(np.uint8)
                    # Map topk indices to colors
                    topk_idx_np = topk_idx.cpu().numpy()
                    # Clip indices to valid range
                    topk_idx_np = np.clip(topk_idx_np, 0, pred_colors_p.shape[0] - 1)
                    occupied_pred_colors = pred_colors_p[topk_idx_np]
                    # if pts_np.shape[0] == occupied_pred_colors.shape[0]:
                    #     vis.scatter(
                    #         X=pts_np,
                    #         win=f"Occupied-points-{filename[p]}",
                    #         opts=dict(
                    #             title=f"Predicted Color - {filename[p]}",
                    #             markercolor=occupied_pred_colors,
                    #             markersize=6
                    #         )
                    #     )
                    
                    all_pred_colors.append(pred_colors[p])
                    all_topk_indices.append(topk_idx)

            # Only save files that contain '13'
            for p in range(B):
                # if '13' in filename[i] or '72' in filename[i]  or '21' in filename[i]: ## 13 filename
                ply_folder = os.path.join(output_dir, "pointclouds-landmarks")
                os.makedirs(ply_folder, exist_ok=True)
                ply_filename = os.path.join(ply_folder, f"{os.path.basename(filename[p])}.ply")
                if args.model == '5' or args.model == '11' or args.model == '10' or args.model == '2' or args.model == '3': # save colors

                    # Get top-k indices for this batch item
                    probs_p = torch.sigmoid(occupancy_pred[p]).squeeze(-1)
                    # top_k_indices[p] is already computed by extract_top_k_occupied_points_val
                    topk_idx = top_k_indices[p]
                    pts_np = pred_points[p][:, [0, 2, 1]].cpu().numpy()  # [top_k, 3]
                    # Get predicted colors for top-k points
                    pred_colors_p = pred_colors[p]  # [N_query, 3]
                    if isinstance(pred_colors_p, torch.Tensor):
                        pred_colors_p = pred_colors_p.detach().cpu().numpy()
                    if pred_colors_p.shape[1] > 3:
                        pred_colors_p = pred_colors_p[:, :3]
                    if pred_colors_p.max() <= 1.0:
                        pred_colors_p = (pred_colors_p * 255).astype(np.uint8)
                    else:
                        pred_colors_p = pred_colors_p.astype(np.uint8)
                    # Map topk indices to colors
                    topk_idx_np = topk_idx.cpu().numpy()
                    # Clip indices to valid range
                    topk_idx_np = np.clip(topk_idx_np, 0, pred_colors_p.shape[0] - 1)
                    occupied_pred_colors = pred_colors_p[topk_idx_np]

                    save_points_as_ply(pts_np, ply_filename, colors=occupied_pred_colors)
                else:
                    save_points_as_ply(pred_points[p], ply_filename)

                # Get species information for this tree
                species_info = ""
                if hasattr(dataset, 'species_list') and dataset.species_list and labels is not None:
                    label = labels[p].item()
                    if label < len(dataset.species_list):
                        species_name = dataset.species_list[label]
                        species_info = f", species: {species_name}"

                print(f"Generated {top_k[p]} occupied points for {filename[p]}, scale: {scale_dsm[p].item()}{species_info}")

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