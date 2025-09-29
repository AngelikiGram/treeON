import torch
import trimesh
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet
import numpy as np

def create_adaptive_query_points(B, gt_mesh, num_query_points=15000, near_tree_ratio=0.85):
    """
    Generate query points by focusing more on regions near the tree surface.

    - near_tree_ratio: fraction of points sampled near tree (default 80%).
    """
    device = gt_mesh.device
    num_near_tree = int(num_query_points * near_tree_ratio)
    num_random = num_query_points - num_near_tree

    # Step 1: Sample **near tree points** using Gaussian noise around GT mesh
    tree_samples = gt_mesh[:, torch.randint(0, gt_mesh.shape[1], (num_near_tree,), device=device)]  # (B, num_near_tree, 3)
    noise = (torch.rand_like(tree_samples) - 0.5) * 0.1 # 0.05 # 0.1 # 0.01 # 1 # 0.05  # Small perturbations
    tree_samples = tree_samples + noise  # Slightly perturb near-tree samples

    # Step 2: Sample **remaining points randomly** in full space
    min_bounds, _ = gt_mesh.min(dim=1, keepdim=True)  # (B, 1, 3)
    max_bounds, _ = gt_mesh.max(dim=1, keepdim=True)  # (B, 1, 3)
    
    random_samples = torch.rand((B, num_random, 3), device=device) * (max_bounds - min_bounds) + min_bounds  # Uniformly in 3D box

    # Step 3: Combine near-tree and random samples
    query_points = torch.cat([tree_samples, random_samples], dim=1)  # Shape: (B, num_query_points, 3)

    return query_points

def relax_query_points(query_points, iterations=5, repulsion_strength=0.1):
    """
    Apply simple point relaxation to spread out points more evenly.

    Args:
        query_points: (B, N, 3) tensor of query points
        iterations: number of relaxation steps
        repulsion_strength: movement step size toward separation

    Returns:
        (B, N, 3) relaxed points
    """
    B, N, _ = query_points.shape
    device = query_points.device

    for _ in range(iterations):
        for b in range(B):
            dists = torch.cdist(query_points[b], query_points[b], p=2)  # (N, N)
            mask = torch.eye(N, device=device).bool()
            dists[mask] = float('inf')  # ignore self-distance

            # Find nearest neighbor for each point
            nn_idx = dists.argmin(dim=-1)  # (N,)
            nn_points = query_points[b, nn_idx]  # (N, 3)

            # Move slightly away from nearest neighbor
            delta = query_points[b] - nn_points
            query_points[b] = query_points[b] + repulsion_strength * delta

    return query_points

def sample_points_in_convex_hull(B, batch_points, num_query_points, device='cuda'):
    B = batch_points.shape[0]
    sampled_all = []

    for i in range(B):
        points_np = batch_points[i].cpu().numpy()
        hull = ConvexHull(points_np)
        delaunay = Delaunay(points_np[hull.vertices])

        samples = []
        count = 0
        while count < num_query_points:
            random_pts = np.random.uniform(
                low=points_np.min(axis=0),
                high=points_np.max(axis=0),
                size=(num_query_points, 3)
            )
            mask = delaunay.find_simplex(random_pts) >= 0
            valid = random_pts[mask]
            samples.append(valid)
            count += len(valid)

        samples_np = np.vstack(samples)[:num_query_points]
        samples_tensor = torch.tensor(samples_np, dtype=batch_points.dtype, device=device)
        sampled_all.append(samples_tensor)

    return torch.stack(sampled_all, dim=0)  # (B, num_query_points, 3)

def sample_points_inside_mesh(B, mesh, num_query_points=15000, device='cuda'):
    """
    Sample points inside the convex hull of each mesh in the batch.

    Args:
        B (int): batch size
        mesh (torch.Tensor): (B, N, 3) input point clouds
        num_query_points (int): number of interior points to sample per mesh
        device (str): output device

    Returns:
        torch.Tensor: (B, num_query_points, 3)
    """
    sampled_all = []

    for i in range(B):
        points_np = mesh[i].cpu().numpy()
        try:
            hull_pts = points_np  # assume full cloud forms convex hull
            delaunay = Delaunay(hull_pts)

            samples = []
            count = 0
            min_bounds = points_np.min(axis=0)
            max_bounds = points_np.max(axis=0)

            while count < num_query_points:
                random_pts = np.random.uniform(
                    low=min_bounds,
                    high=max_bounds,
                    size=(num_query_points, 3)
                )
                mask = delaunay.find_simplex(random_pts) >= 0
                valid = random_pts[mask]
                samples.append(valid)
                count += len(valid)

            samples_np = np.vstack(samples)[:num_query_points]
            sampled_tensor = torch.tensor(samples_np, dtype=mesh.dtype, device=device)
            sampled_all.append(sampled_tensor)
        except Exception as e:
            print(f"Skipping batch {i} due to Delaunay error: {e}")
            sampled_all.append(torch.zeros(num_query_points, 3, dtype=mesh.dtype, device=device))

    return torch.stack(sampled_all, dim=0)  # (B, num_query_points, 3)

def sample_points_inside_mesh1(B, mesh, num_query_points=15000, jitter_std=0.01, device='cuda'): # jitter_std=0.02
    """
    Quickly sample points near/in the point cloud by jittering real points.

    Args:
        mesh (torch.Tensor): (B, N, 3) point cloud batch
        num_query_points (int): number of points to sample per batch item
        jitter_std (float): standard deviation of the noise
        device (str): device for output

    Returns:
        torch.Tensor: (B, num_query_points, 3)
    """
    B, N, _ = mesh.shape

    # Randomly sample indices
    indices = torch.randint(0, N, (B, num_query_points), device=device)
    batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, num_query_points)

    # Gather base points
    base_points = mesh[batch_indices, indices]  # (B, num_query_points, 3)

    # Add jitter noise
    noise = torch.randn_like(base_points) * jitter_std
    return base_points + noise

def create_normalized_query_points1(B, num_query_points=15000, device='cuda'):
    """
    Create query points uniformly sampled in the unit cube [0, 1]^3.
    
    Args:
        B (int): Batch size.
        num_query_points (int): Number of query points per batch.
        device (str): Device to create the tensor on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Query points of shape (B, num_query_points, 3) in [0, 1]^3.
    """
    return torch.rand((B, num_query_points, 3), device=device)


def create_normalized_query_points(B: int,
                           num_query_points: int,
                           device: str = "cuda",
                           engine_seed: int | None = None) -> torch.Tensor:
    """
    DSM-agnostic sampler: scrambled Sobol in [0,1]^3.
    - Excellent uniformity & projection properties (better than iid uniform).
    - Independent Owen-scramble per batch (via different seeds).

    Args:
        B: batch size
        num_query_points: points per batch
        device: 'cuda' or 'cpu'
        engine_seed: base seed for reproducibility (None => random)

    Returns:
        Tensor of shape (B, num_query_points, 3) in [0,1].
    """
    pts_cpu = []
    for b in range(B):
        seed = None if engine_seed is None else engine_seed + b
        eng = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
        pts_b = eng.draw(num_query_points)   # (N, 3) on CPU, float32 in [0,1)
        pts_cpu.append(pts_b)
    pts = torch.stack(pts_cpu, dim=0).to(device)  # (B, N, 3)
    return pts

# def create_normalized_query_points(
#     B: int,
#     num_query_points: int = 15_000,
#     device: str = "cuda",
#     method: str = "sobol",
#     jitter: float = 0.5,
# ):
#     """
#     Create query points in [0,1]^3 using more sophisticated sampling.

#     Args:
#         B: batch size
#         num_query_points: number of points per batch
#         device: 'cuda' or 'cpu'
#         method: one of {'sobol', 'stratified', 'lhs'}
#         jitter: controls intra-cell jitter for 'stratified' (0=no jitter, 0.5=mid cell +/- half-cell)

#     Returns:
#         (B, num_query_points, 3) tensor on `device`, values in [0,1].
#     """
#     method = method.lower()
#     if method not in {"sobol", "stratified", "lhs"}:
#         raise ValueError("method must be one of {'sobol','stratified','lhs'}")

#     if method == "sobol":
#         # Low-discrepancy sequence with very uniform coverage and good projection properties.
#         # Generated on CPU (as required by torch SobolEngine) and moved to device.
#         engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
#         pts = engine.draw(num_query_points)  # (N, 3) in [0,1)
#         pts = pts.to(device)

#     elif method == "stratified":
#         # Partition the cube into a near-cubic grid, sample one point per cell (optionally jittered).
#         n_per_axis = math.ceil(num_query_points ** (1/3))
#         # Build cell centers
#         lin = torch.linspace(0, 1, steps=n_per_axis + 1, device=device)  # cell edges
#         # cell centers (exclude last edge)
#         edges = lin[:-1]
#         cell_size = 1.0 / n_per_axis
#         centers = edges + 0.5 * cell_size

#         gx, gy, gz = torch.meshgrid(centers, centers, centers, indexing='ij')
#         grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (n_per_axis^3, 3)

#         if jitter > 0:
#             # jitter uniformly within each cell, limited by `jitter` fraction of half-cell
#             max_offset = (cell_size * jitter)
#             grid = grid + (torch.rand_like(grid) * 2 - 1) * max_offset
#             grid.clamp_(0.0, 1.0)

#         pts = grid[:num_query_points]  # trim excess

#     else:  # method == 'lhs'
#         # Latin Hypercube: ensures one sample per stratum along each axis
#         n = num_query_points
#         # strata edges per axis
#         strata = (torch.rand(n, 3, device=device) + torch.arange(n, device=device).unsqueeze(1)) / n  # (n,3)
#         # independent random permutations per axis
#         perm_x = torch.randperm(n, device=device)
#         perm_y = torch.randperm(n, device=device)
#         perm_z = torch.randperm(n, device=device)
#         pts = torch.stack([strata[perm_x, 0], strata[perm_y, 1], strata[perm_z, 2]], dim=1)
#         # clamp (should already be within [0,1))
#         pts = pts.clamp_(0.0, 1.0)

#     # Tile for batch
#     pts = pts.unsqueeze(0).expand(B, -1, -1).contiguous()
#     return pts

# def create_normalized_query_points(B, num_query_points=15000, device='cuda'):
#     """
#     Create query points uniformly sampled in the unit cube [0, 1]^2 x [0, 1.2] in z.
    
#     Args:
#         B (int): Batch size.
#         num_query_points (int): Number of query points per batch.
#         device (str): Device to create the tensor on (e.g., 'cuda' or 'cpu').

#     Returns:
#         torch.Tensor: Query points of shape (B, num_query_points, 3) in [0, 1]^2 x [0, 1.2].
#     """
#     xy = torch.rand((B, num_query_points, 2), device=device)
#     z = torch.rand((B, num_query_points, 1), device=device) * 1.2
#     return torch.cat([xy, z], dim=-1)

def create_query_points(B, gt_mesh, num_query_points=15000):
    # Compute min and max separately for X, Y, Z
    min_gt_x = gt_mesh[:, :, 0].min(dim=1, keepdim=True)[0]  # (B, 1)
    min_gt_y = gt_mesh[:, :, 1].min(dim=1, keepdim=True)[0]  # (B, 1)
    min_gt_z = gt_mesh[:, :, 2].min(dim=1, keepdim=True)[0]  # (B, 1)

    max_gt_x = gt_mesh[:, :, 0].max(dim=1, keepdim=True)[0]  # (B, 1)
    max_gt_y = gt_mesh[:, :, 1].max(dim=1, keepdim=True)[0]  # (B, 1)
    max_gt_z = gt_mesh[:, :, 2].max(dim=1, keepdim=True)[0]  # (B, 1)

    # Expand min/max values to match the query shape (B, 15000, 1)
    min_gt_x, min_gt_y, min_gt_z = [v.unsqueeze(1).expand(B, num_query_points, 1) for v in [min_gt_x, min_gt_y, min_gt_z]]
    max_gt_x, max_gt_y, max_gt_z = [v.unsqueeze(1).expand(B, num_query_points, 1) for v in [max_gt_x, max_gt_y, max_gt_z]]

    # Sample query points in [0,1] range, then scale per dimension
    query_x = torch.rand((B, num_query_points, 1), device=gt_mesh.device) * (max_gt_x - min_gt_x) + min_gt_x
    query_y = torch.rand((B, num_query_points, 1), device=gt_mesh.device) * (max_gt_y - min_gt_y) + min_gt_y
    query_z = torch.rand((B, num_query_points, 1), device=gt_mesh.device) * (max_gt_z - min_gt_z) + min_gt_z

    # Concatenate into (B, 15000, 3)
    query_points = torch.cat([query_x, query_y, query_z], dim=-1)  # Shape: (B, 15000, 3)

    return query_points # relax_query_points(query_points)


# -------------------------
# -------------------------
# -------------------------

def create_query_points_best(B, gt_mesh, num_query_points=15000, oversample_factor=4):
    """
    Generate query points using FPS on uniformly sampled points from bounding box.

    Args:
        gt_mesh: (B, N_gt, 3) GT point cloud
        num_query_points: final number of query points
        oversample_factor: how many points to generate before FPS

    Returns:
        (B, num_query_points, 3)
    """
    B = gt_mesh.shape[0]
    N_dense = oversample_factor * num_query_points

    # Get bounding box
    min_xyz = gt_mesh.min(dim=1, keepdim=True)[0]
    max_xyz = gt_mesh.max(dim=1, keepdim=True)[0]

    # Sample dense points in bbox
    dense_points = torch.rand(B, N_dense, 3, device=gt_mesh.device)
    dense_points = dense_points * (max_xyz - min_xyz) + min_xyz

    # Apply FPS to get well-spaced samples
    query_points = farthest_point_sampling(dense_points, num_query_points)
    return query_points

def farthest_point_sampling(xyz, n_samples):
    """
    Pure PyTorch implementation of Farthest Point Sampling (FPS).

    Args:
        xyz: (B, N, 3) input point cloud
        n_samples: number of samples to return

    Returns:
        fps_points: (B, n_samples, 3)
    """
    B, N, _ = xyz.shape
    fps_points = torch.zeros((B, n_samples, 3), device=xyz.device)
    indices = torch.zeros((B, n_samples), dtype=torch.long, device=xyz.device)

    # Initialize with a random point per batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    dist = torch.full((B, N), float('inf'), device=xyz.device)

    for i in range(n_samples):
        indices[:, i] = farthest
        fps_points[:, i] = xyz[batch_indices, farthest]

        dist_to_new = torch.norm(xyz - fps_points[:, i].unsqueeze(1), dim=-1)
        dist = torch.min(dist, dist_to_new)
        farthest = torch.max(dist, dim=1)[1]

    return fps_points
