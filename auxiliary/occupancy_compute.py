import torch
import torch.nn.functional as F
import numpy as np

def redistribute_dsm_points(dsm_pc):
    """
    Randomly redistributes DSM points across the bounding box to remove spatial bias.
    
    Args:
        dsm_pc: Tensor of shape (B, P, 3)

    Returns:
        redistributed_dsm_pc: Tensor of same shape with shuffled positions
    """
    B, P, _ = dsm_pc.shape
    device = dsm_pc.device
    redistributed = []

    for b in range(B):
        min_bound = dsm_pc[b].min(dim=0)[0]  # (3,)
        max_bound = dsm_pc[b].max(dim=0)[0]  # (3,)

        random_points = torch.rand((P, 3), device=device) * (max_bound - min_bound) + min_bound
        redistributed.append(random_points.unsqueeze(0))

    return torch.cat(redistributed, dim=0)  # (B, P, 3)

def estimate_average_spacing(query_points, num_samples=1000):
    """
    Estimate average pairwise spacing by sampling point pairs.
    Args:
        query_points: (B, N, 3)
        num_samples: number of pairs to sample (default: 1000)
    Returns:
        avg_spacing: (B,) estimated average distance per batch
    """
    B, N, _ = query_points.shape
    device = query_points.device
    avg_spacing = []

    for b in range(B):
        idx1 = torch.randint(0, N, (num_samples,), device=device)
        idx2 = torch.randint(0, N, (num_samples,), device=device)
        dist = torch.norm(query_points[b, idx1] - query_points[b, idx2], dim=-1)
        avg_spacing.append(dist.mean())

    return torch.stack(avg_spacing)  # (B,)

def compute_occupancy(B, custom_query_points, gt_custom_mask, top_k=1000, threshold=20):
    """
    Computes shadow occupancy based on the ground-truth shadow mask.
    """
    gt_custom_occupancy = torch.zeros((B, custom_query_points.shape[1], 1), device=custom_query_points.device)

    threshold = estimate_average_spacing(custom_query_points, num_samples=1000)[0] / threshold

    for b in range(B):
        # Compute pairwise distances between all query points
        # dist_matrix = torch.cdist(custom_query_points[b], custom_query_points[b])  # (N, N)
        # mean_dist = dist_matrix.mean()
        # threshold = mean_dist / threshold

        # Compute min distances from query points to shadow mask
        distances = torch.cdist(custom_query_points[b], gt_custom_mask[b])  # (N, M)
        min_distances, _ = distances.min(dim=-1)  # (N,)
        gt_custom_occupancy[b] = (min_distances <= threshold).float().unsqueeze(-1)

    return gt_custom_occupancy
    # gt_custom_occupancy = torch.zeros((B, custom_query_points.shape[1], 1), device=custom_query_points.device)
    # for b in range(B):
    #     # Check if each query point is inside the shadow
    #     distances = torch.cdist(custom_query_points[b], gt_custom_mask[b])
    #     min_distances, _ = distances.min(dim=-1)
    #     threshold_value = torch.kthvalue(min_distances, top_k).values
    #     gt_custom_occupancy[b] = (min_distances <= threshold_value).float().unsqueeze(-1)
    # return gt_custom_occupancy

def compute_occupancy_top_k(B, custom_query_points, gt_custom_mask, top_k=1000):
    """
    Computes shadow occupancy using the top_k nearest points in the shadow mask.
    """
    gt_custom_occupancy = torch.zeros((B, custom_query_points.shape[1], 1), device=custom_query_points.device)

    for b in range(B):
        # Compute distances from query points to shadow mask
        distances = torch.cdist(custom_query_points[b], gt_custom_mask[b])  # (N, M)
        min_distances, _ = distances.min(dim=-1)  # (N,)

        # Determine threshold based on top_k smallest distances
        threshold_value = torch.kthvalue(min_distances, top_k).values
        gt_custom_occupancy[b] = (min_distances <= threshold_value).float().unsqueeze(-1)

    return gt_custom_occupancy

# ===========================
# Extract **Occupied Tree and Shadow Points**
# ===========================

def extract_threshold_occupied_points(occupancy_pred, query_points, threshold=0.5, max_points=None):
    """
    Extracts points where occupancy > threshold.
    Optionally limits to max_points.
    """
    occupied_points = []
    B = query_points.shape[0]

    for b in range(B):
        mask = occupancy_pred[b].squeeze(-1) > threshold  # Shape: (N,)
        points = query_points[b][mask]  # (K, 3)

        # Optional: limit number of points
        if max_points is not None and points.shape[0] > max_points:
            idx = torch.randperm(points.shape[0])[:max_points]
            points = points[idx]

        occupied_points.append(points)

    return occupied_points

def extract_top_k_occupied_points_1(occupancy_pred, query_points, top_k=1000):
    '''
    Extracts the top-k occupied points from the occupancy predictions (keeps gradients).
    '''
    weights = torch.sigmoid(occupancy_pred).squeeze(-1)  # (B, N)
    weights = weights.clamp(min=1e-3)  # Prevent vanishing gradients

    topk_vals, topk_idx = torch.topk(weights, top_k, dim=1)

    B = query_points.shape[0]
    topk_points = []
    topk_weights = []

    for b in range(B):
        pts = query_points[b, topk_idx[b]]           # (top_k, 3)
        wts = topk_vals[b]                           # (top_k,)

        topk_points.append(pts)
        topk_weights.append(wts)

    return topk_points


def extract_top_k_occupied_points_val(occupancy_pred, query_points, top_k_tensor):
    '''
    Extracts the top-k occupied points from the occupancy predictions (keeps gradients).
    top_k_tensor: Tensor of shape (B,) with individual top_k values per batch item
    '''
    weights = torch.sigmoid(occupancy_pred).squeeze(-1)  # (B, N)
    weights = weights.clamp(min=1e-3)  # Prevent vanishing gradients

    B = query_points.shape[0]
    topk_points = []
    topk_weights = []
    topk_indices = []

    for b in range(B):
        k = top_k_tensor[b].item()
        topk_vals, topk_idx = torch.topk(weights[b], k)

        pts = query_points[b, topk_idx]  # (k, 3)
        wts = topk_vals                 # (k,)

        topk_points.append(pts)
        topk_weights.append(wts)
        topk_indices.append(topk_idx)

    return topk_points, topk_indices

def extract_top_k_occupied_points(occupancy_pred, query_points, top_k=1000):
    '''
    Extracts the top-k occupied points from the occupancy predictions (keeps gradients).
    `top_k` can be an int or a tensor of shape (B,) for per-batch top-k values.
    '''
    weights = torch.sigmoid(occupancy_pred).squeeze(-1)  # (B, N)
    weights = weights.clamp(min=1e-3)  # Prevent vanishing gradients

    B, N = weights.shape
    topk_points = []
    topk_indices = []

    # If top_k is a scalar int, expand it into a tensor
    if isinstance(top_k, int):
        top_k = torch.full((B,), top_k, dtype=torch.long, device=weights.device)
    elif isinstance(top_k, list):
        top_k = torch.tensor(top_k, dtype=torch.long, device=weights.device)
    else:
        assert isinstance(top_k, torch.Tensor) and top_k.shape[0] == B

    for b in range(B):
        k = min(top_k[b].item(), N)
        topk_vals, topk_idx = torch.topk(weights[b], k)
        pts = query_points[b, topk_idx]  # (k, 3)

        topk_points.append(pts)
        topk_indices.append(topk_idx)

    return topk_points, topk_indices

def extract_top_k_occupied_pointsOLD(occupancy_pred, query_points, top_k=1000):
    '''
    Extracts the top-k occupied points from the occupancy predictions (keeps gradients).
    '''
    weights = torch.sigmoid(occupancy_pred).squeeze(-1)  # (B, N)
    weights = weights.clamp(min=1e-3)  # Prevent vanishing gradients

    topk_vals, topk_idx = torch.topk(weights, top_k, dim=1)

    B = query_points.shape[0]
    topk_points = []
    topk_weights = []

    for b in range(B):
        pts = query_points[b, topk_idx[b]]           # (top_k, 3)
        wts = topk_vals[b]                           # (top_k,)

        topk_points.append(pts)
        topk_weights.append(wts)

    return topk_points, topk_idx

def extract_soft_top_occupied_points(occupancy_pred, query_points, top_k=1000, sharpness=50.0, noise_std=1e-3):
    """
    Fully differentiable approximation of top-k occupied points using softmax-based attention.
    Returns a (B, top_k, 3) tensor of softly selected points.
    """
    B, N, _ = query_points.shape

    # Compute sharpened scores
    scores = torch.softmax(occupancy_pred.squeeze(-1) * sharpness, dim=1)  # (B, N)

    # Repeat scores for top_k "slots"
    soft_weights = scores.unsqueeze(1).expand(B, top_k, N)  # (B, top_k, N)

    # Add small noise to encourage diversity (optional)
    if noise_std > 0:
        noise = torch.randn_like(soft_weights) * noise_std
        soft_weights = soft_weights + noise

    # Normalize weights per slot
    soft_weights = torch.softmax(soft_weights * sharpness, dim=2)  # (B, top_k, N)

    # Soft selection: attention over query_points
    selected = torch.bmm(soft_weights, query_points)  # (B, top_k, 3)

    return selected


def pad_or_trim(points_list, target_k=1000):
    """
    Pads or trims a list of (N_i, 3) tensors to (B, target_k, 3)
    """
    padded = []
    for pts in points_list:
        N = pts.shape[0]
        if N >= target_k:
            idx = torch.randperm(N)[:target_k]
            padded_pts = pts[idx]
        else:
            pad = torch.zeros((target_k - N, 3), dtype=pts.dtype, device=pts.device)
            padded_pts = torch.cat([pts, pad], dim=0)
        padded.append(padded_pts.unsqueeze(0))
    return torch.cat(padded, dim=0)  # (B, target_k, 3)

from scipy.spatial import Delaunay

def redistribute_points_in_volume(batch_points: torch.Tensor, num_samples: int = None) -> torch.Tensor:
    """
    Redistribute each batch of 3D points inside its own convex hull using rejection sampling.

    Args:
        batch_points: (B, N, 3) tensor
        num_samples: number of points to generate per batch item

    Returns:
        (B, num_samples, 3) tensor of redistributed points
    """
    B, N, _ = batch_points.shape
    if num_samples is None:
        num_samples = N

    redistributed = []

    for b in range(B):
        points_np = batch_points[b].cpu().numpy()
        hull = Delaunay(points_np)

        min_bounds = points_np.min(axis=0)
        max_bounds = points_np.max(axis=0)

        samples = []
        tries = 0
        max_tries = num_samples * 50

        while len(samples) < num_samples and tries < max_tries:
            sample = np.random.uniform(min_bounds, max_bounds)
            if hull.find_simplex(sample) >= 0:
                samples.append(sample)
            tries += 1

        if len(samples) < num_samples:
            print(f"Warning: batch {b} only got {len(samples)} samples (needed {num_samples})")

        samples_np = np.array(samples, dtype=np.float32)
        redistributed.append(torch.tensor(samples_np, device=batch_points.device).unsqueeze(0))

    return torch.cat(redistributed, dim=0)  # (B, num_samples, 3)