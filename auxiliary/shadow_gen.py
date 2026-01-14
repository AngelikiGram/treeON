import torch

def rotate_mesh_z_90(mesh):
    """
    Rotates the mesh 90 degrees around the Z-axis.
    mesh: (B, P, 3)
    """
    B, P, _ = mesh.shape
    rot_z_90 = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=mesh.device)

    return torch.matmul(mesh, rot_z_90.T)  # (B, P, 3)

def rotate_mesh_y_90(mesh):
    """
    Rotates the mesh 90 degrees around the Y-axis, centered on the mesh.
    mesh: (B, P, 3)
    """
    # Compute the center of the mesh
    center = mesh.mean(dim=1, keepdim=True)  # (B, 1, 3)

    # Shift mesh to origin
    mesh_centered = mesh - center  # (B, P, 3)

    # Define 90 degree rotation matrix around Y axis
    rot_y_90 = torch.tensor([
        [0, 0, -1],
        [0, 1,  0],
        [1, 0,  0]
    ], dtype=mesh.dtype, device=mesh.device)  # shape (3, 3)

    # Apply rotation
    rotated = torch.matmul(mesh_centered, rot_y_90.T)  # (B, P, 3)

    # Translate back to original center
    rotated = rotated + center

    return rotated  # (B, P, 3)

# ------------------------------------------------
def flip_mesh_x(mesh):
    """
    Flips the mesh across the X-axis (left ↔ right mirror).
    mesh: Tensor of shape (B, P, 3)
    """
    flipped = mesh.clone()
    flipped[..., 0] = -flipped[..., 0]  # Invert X-coordinates
    return flipped
def flip_mesh_y(mesh):
    """
    Flips the mesh across the Y-axis (front ↔ back mirror).
    mesh: Tensor of shape (B, P, 3)
    """
    flipped = mesh.clone()
    flipped[..., 1] = -flipped[..., 1]  # Invert Y-coordinates
    return flipped
def flip_mesh_z(mesh):
    """
    Flips the mesh across the Z-axis (up ↔ down mirror).
    mesh: Tensor of shape (B, P, 3)
    """
    flipped = mesh.clone()
    flipped[..., 2] = -flipped[..., 2]  # Invert Z-coordinates
    return flipped
def rotate_mesh_z(mesh, angle_deg):
    """
    Rotates the mesh around the Z-axis by the given angle in degrees.
    mesh: (B, P, 3)
    """
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0, dtype=mesh.dtype, device=mesh.device)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

    rot_z = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=mesh.dtype, device=mesh.device)

    center = mesh.mean(dim=1, keepdim=True)
    mesh_centered = mesh - center
    rotated = torch.matmul(mesh_centered, rot_z.T)
    return rotated + center

def softmin(x, dim=0, temperature=10.0):
    weights = torch.nn.functional.softmax(-x * temperature, dim=dim)
    return (x * weights).sum(dim=dim, keepdim=True)

def generate_shadow(vis, mesh, light_dir_batch, temperature=10.0, eps=1e-6):
    """
    Projects 3D mesh onto XY plane using light direction, safely handling negative Z directions.
    Ensures differentiable gradients flow properly through the projection.
    """
    mesh = rotate_mesh_x_90(mesh)
    B, P, _ = mesh.shape

    # Normalize light direction
    light_dir_batch = light_dir_batch.view(B, 1, 3)
    light_dir_batch = light_dir_batch / (torch.norm(light_dir_batch, dim=2, keepdim=True) + eps)

    # # Flip light direction if pointing upward (Z < 0)
    # light_dir_batch = torch.where(light_dir_batch[:, :, 2:3] < 0, -light_dir_batch, light_dir_batch)

    # Safe t-scalar calculation: project towards Z = 0
    light_dir_z = light_dir_batch[:, :, 2] + eps  # Now guaranteed positive
    t = -mesh[:, :, 2] / light_dir_z             # No division by zero, stable gradients

    # Project points along the light direction
    shadow_points = mesh + t.unsqueeze(2) * light_dir_batch


    # ------------------------------
    
    # --- Extract XY shadow points (Z is now 0 after projection)
    shadow_xy = shadow_points[:, :, :2]  # shape: (B, P, 2)
    P = shadow_xy.shape[1]

    # --- Center points
    shadow_centered = shadow_xy - shadow_xy.mean(dim=1, keepdim=True)

    # PCA + Swap + X-shift
    eps = 1e-6
    pca_aligned = []
    eye = torch.eye(2, device=shadow_xy.device, dtype=shadow_xy.dtype)

    for b in range(B):
        # Covariance with regularization
        cov = shadow_centered[b].T @ shadow_centered[b] / P
        cov = cov + eps * eye  # makes eigvals distinct, prevents instability

        # Eigen-decomposition of symmetric matrix
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Rotate so major component comes last (Y axis)
        rotation = eigvecs.flip(dims=[1])  # [major, minor]

        rotated = shadow_centered[b] @ rotation
        rotated = rotated[:, [1, 0]]  # Swap so major → Y

        # Shift into [0, 1]^2
        rotated += 0.5

        pca_aligned.append(rotated)

    pca_aligned = torch.stack(pca_aligned, dim=0)

    # Back to 3D
    shadow_points = torch.cat([
        pca_aligned[..., 0:1],  # X
        pca_aligned[..., 1:2],  # Y
        torch.zeros_like(pca_aligned[..., 0:1])  # Z = 0
    ], dim=-1)
    
    # ------------------------------

    return shadow_points

def rotate_mesh_z_180(mesh):
    """
    Rotates the mesh 180 degrees around the Z-axis.
    mesh: (B, P, 3)
    """
    B, P, _ = mesh.shape
    rot_z_180 = torch.tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=mesh.device)

    return torch.matmul(mesh, rot_z_180.T)  # (B, P, 3)
def rotate_mesh_x_90(mesh):
    """
    Rotates the mesh 90 degrees around the X-axis.
    mesh: (B, P, 3)
    """
    B, P, _ = mesh.shape
    rot_x_90 = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=torch.float32, device=mesh.device)

    return torch.matmul(mesh, rot_x_90.T)  # (B, P, 3)
def soft_bounds(xy, temperature=10.0):
    # Normalize xy before softmax to avoid extreme values
    xy_centered = xy - xy.mean(dim=1, keepdim=True)
    scale = xy_centered.abs().max(dim=1, keepdim=True)[0] + 1e-6
    xy_normalized = xy_centered / scale

    weights_min = torch.softmax(-xy_normalized * temperature, dim=1)
    weights_max = torch.softmax(xy_normalized * temperature, dim=1)

    soft_min = (xy * weights_min).sum(dim=1, keepdim=True)
    soft_max = (xy * weights_max).sum(dim=1, keepdim=True)

    return soft_min, soft_max


def normalize_to_unit_box_preserving_aspect_soft(xy, temperature=5.0, eps=1e-6, border=0.05):
    """
    Fully differentiable normalization to [border, 1-border]^2.
    Preserves aspect ratio and fits the entire shape within the box.
    """
    # Step 1: Soft bounds
    soft_min, soft_max = soft_bounds(xy, temperature=temperature)  # (B, 1, 2)
    center = (soft_max + soft_min) / 2                             # (B, 1, 2)
    extent = soft_max - soft_min                                  # (B, 1, 2)

    # Step 2: Uniform scaling factor based on largest axis
    # scale = extent.max(dim=2, keepdim=True)[0] + eps              # (B, 1, 1)
    scale = extent.max(dim=2, keepdim=True)[0].clamp(min=1e-2)

    # Step 3: Center and scale
    xy_centered = xy - center                                     # (B, N, 2)
    normalized = (xy_centered / scale) + 0.5                      # now in [0, 1]

    # Step 4: Apply border margin
    margin = border
    normalized = normalized * (1 - 2 * margin) + margin           # now in [border, 1-border]

    return normalized

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch.nn.functional as F

def differentiable_crop(shadow, output_size=124, threshold=0.05):
    """
    Crop to the white (high activation) region tightly and resize.
    This returns a differentiable crop based on soft bounding box.
    
    shadow: (B, 1, H, W)
    Returns: (B, 1, output_size, output_size)
    """
    B, _, H, W = shadow.shape
    device = shadow.device

    output_size1 = int(output_size) # // 2 + 15)  

    # Normalize shadow to [0, 1]
    norm_shadow = shadow / (shadow.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
    # norm_shadow = shadow - shadow.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    # norm_shadow = norm_shadow / (norm_shadow.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)

    # Threshold mask (soft, keep it differentiable)
    mask = (norm_shadow > threshold).float()  # (B, 1, H, W)

    # Create coordinate grids in [0, 1]
    y_coords = torch.linspace(0, 1, H, device=device)
    x_coords = torch.linspace(0, 1, W, device=device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    X = X.view(1, 1, H, W)
    Y = Y.view(1, 1, H, W)

    # Masked coordinate mins and maxes
    def masked_min(tensor, mask):
        masked = tensor + (1.0 - mask) * 1e6  # push masked-out values high
        return masked.view(B, -1).min(dim=1)[0]

    def masked_max(tensor, mask):
        masked = tensor - (1.0 - mask) * 1e6  # push masked-out values low
        return masked.view(B, -1).max(dim=1)[0]

    x_min = masked_min(X, mask)
    x_max = masked_max(X, mask)
    y_min = masked_min(Y, mask)
    y_max = masked_max(Y, mask)

    # Convert crop coords to [-1, 1] range
    grids = []
    for b in range(B):
        lin_x = torch.linspace(x_min[b], x_max[b], output_size, device=device) * 2 - 1
        lin_y = torch.linspace(y_min[b], y_max[b], output_size1, device=device) * 2 - 1
        grid_y, grid_x = torch.meshgrid(lin_y, lin_x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grids.append(grid)

    grid = torch.stack(grids, dim=0)  # (B, H, W, 2)
    cropped = F.grid_sample(shadow, grid, align_corners=True)

    return cropped

def soft_bounds(x, temperature=10.0):
    """
    Computes differentiable approximations of min and max using softmin/softmax.
    x: (N, D)
    Returns:
        soft_min, soft_max: each of shape (1, D)
    """
    soft_min = torch.sum(x * torch.softmax(-x * temperature, dim=0), dim=0, keepdim=True)
    soft_max = torch.sum(x * torch.softmax(x * temperature, dim=0), dim=0, keepdim=True)
    return soft_min, soft_max

def differentiable_point_projection(points, image_size=124, sigma=1.0, output_size=124, weights=None, axis='y', shadowBoolean=False, temperature=5.0, eps=0.5):
    """
    Differentiable rendering of point clouds into 2D images (projected shadow),
    followed by differentiable cropping to keep only the white region.
    """
    result = []
    for b in range(len(points)):
        pts = points[b][..., :2] if points[b].shape[-1] == 3 else points[b]  # (N, 2)

        if axis == 'x':
            pts = points[b][..., [1, 2]]
        elif axis == 'z':
            pts = points[b][..., [0, 2]]

        # soft_min, soft_max = soft_bounds(pts, temperature=1.0)
        # extent = soft_max - soft_min + 1e-6  # prevent div by zero
        # pts = (pts - soft_min) / extent

        N = pts.shape[0]
        coord = pts * (image_size - 1)

        # Create 2D grid of coordinates (image space)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(image_size, device=pts.device),
            torch.arange(image_size, device=pts.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float().view(1, image_size, image_size, 2)  # (1, H, W, 2)

        # Compute Gaussian heatmap for each point
        dist = torch.sum((grid - coord.view(N, 1, 1, 2)) ** 2, dim=-1)  # (N, H, W)
        heatmap = torch.exp(-dist / (2 * sigma ** 2))  # (N, H, W)

        if weights is not None:
            wts = weights[b].view(N, 1, 1)
            heatmap = heatmap * wts

        # Sum all Gaussians into a single shadow map
        # shadow = heatmap.sum(dim=0, keepdim=True) # .clamp(0.0, 1.0)  # (1, H, W)

        # Option 1: smooth saturation
        shadow = heatmap.sum(dim=0, keepdim=True)
        shadow = shadow / (shadow + 1)

        # Convert to (B, 1, H, W) and crop differentiably
        shadow_batched = shadow.unsqueeze(0)  # (1, 1, H, W)
        resized = shadow_batched # differentiable_crop(shadow_batched, output_size=output_size)  # (1, 1, output_size, output_size)

        if shadowBoolean: 
            resized = differentiable_crop(shadow_batched, output_size=output_size)

        result.append(resized.squeeze(0))  # (1, H, W)

    return torch.stack(result, dim=0)  # (B, 1, H, W)

def soft_bounds(x, temperature=10.0):
    """
    Differentiable approximation of min and max per column.
    """
    soft_min = torch.sum(x * torch.softmax(-x * temperature, dim=0), dim=0, keepdim=True)
    soft_max = torch.sum(x * torch.softmax(x * temperature, dim=0), dim=0, keepdim=True)
    return soft_min, soft_max

def differentiable_point_projectionShadow(points, image_size=124, sigma=1.0, output_size=124,
                                 weights=None, shadowBoolean=False, eps=1e-6, axis='y'):
    """
    Differentiable rendering of point clouds into 2D images (projected shadow),
    followed by differentiable cropping to keep only the white region.
    """
    result = []
    for b in range(len(points)):
        pts = points[b][..., :2] if points[b].shape[-1] == 3 else points[b]  # (N, 2)

        if axis == 'x':
            pts = points[b][..., [1, 2]]
        elif axis == 'z':
            pts = points[b][..., [0, 2]]

        # soft_min, soft_max = soft_bounds(pts, temperature=1.0)
        # extent = soft_max - soft_min + 1e-6  # prevent div by zero
        # pts = (pts - soft_min) / extent

        N = pts.shape[0]
        coord = pts * (image_size - 1)

        # Create 2D grid of coordinates (image space)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(image_size, device=pts.device),
            torch.arange(image_size, device=pts.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float().view(1, image_size, image_size, 2)  # (1, H, W, 2)

        # Compute Gaussian heatmap for each point
        dist = torch.sum((grid - coord.view(N, 1, 1, 2)) ** 2, dim=-1)  # (N, H, W)
        heatmap = torch.exp(-dist / (2 * sigma ** 2))  # (N, H, W)

        if weights is not None:
            wts = weights[b].view(N, 1, 1)
            heatmap = heatmap * wts

        # Sum all Gaussians into a single shadow map
        # shadow = heatmap.sum(dim=0, keepdim=True) # .clamp(0.0, 1.0)  # (1, H, W)

        # Option 1: smooth saturation
        shadow = heatmap.sum(dim=0, keepdim=True)
        shadow = shadow / (shadow + 1)

        # Convert to (B, 1, H, W) and crop differentiably
        shadow_batched = shadow.unsqueeze(0)  # (1, 1, H, W)
        resized = shadow_batched # differentiable_crop(shadow_batched, output_size=output_size)  # (1, 1, output_size, output_size)

        if shadowBoolean: 
            resized = differentiable_crop(shadow_batched, output_size=output_size)

        result.append(resized.squeeze(0))  # (1, H, W)

    return torch.stack(result, dim=0)  # (B, 1, H, W)

def differentiableProjection(points, image_size=124, sigma=1.0, output_size=124,
                                 weights=None, shadowBoolean=False, eps=1e-6, axis='y'):
    """
    Differentiable rendering of point clouds into 2D images (projected shadow),
    followed by differentiable cropping to keep only the white region.
    """
    result = []
    for b in range(len(points)):
        pts = points[b][..., :2] if points[b].shape[-1] == 3 else points[b]  # (N, 2)

        if axis == 'x':
            pts = points[b][..., [1, 2]]
        elif axis == 'z':
            pts = points[b][..., [0, 2]]

        N = pts.shape[0]
        coord = pts * (image_size - 1)

        # # change for the peak
        # margin = 4  # in pixels
        # scale = (image_size - 1 - 2 * margin)
        # coord = pts * scale + margin
        
        ##
        # margin = 4
        # scale = torch.tensor(image_size - 1 - 2 * margin, dtype=pts.dtype, device=pts.device)
        # coord = pts * scale + margin

        # pts_for_range = pts.clone().detach()  
        # min_xy = pts_for_range.min(dim=0, keepdim=True)[0]
        # max_xy = pts_for_range.max(dim=0, keepdim=True)[0]
        # extent = (max_xy - min_xy).clamp(min=1e-6)
        # coord = (pts - min_xy) / extent * (image_size - 1)

        # Create 2D grid of coordinates (image space)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(image_size, device=pts.device),
            torch.arange(image_size, device=pts.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float().view(1, image_size, image_size, 2)  # (1, H, W, 2)

        # Compute Gaussian heatmap for each point
        dist = torch.sum((grid - coord.view(N, 1, 1, 2)) ** 2, dim=-1)  # (N, H, W)
        heatmap = torch.exp(-dist / (2 * sigma ** 2))  # (N, H, W)

        if weights is not None:
            wts = weights[b].view(N, 1, 1)
            heatmap = heatmap * wts

        # Sum all Gaussians into a single shadow map
        # shadow = heatmap.sum(dim=0, keepdim=True) # .clamp(0.0, 1.0)  # (1, H, W)

        # Option 1: smooth saturation
        shadow = heatmap.sum(dim=0, keepdim=True)
        shadow = shadow / (shadow + 1)

        # Convert to (B, 1, H, W) and crop differentiably
        shadow_batched = shadow.unsqueeze(0)  # (1, 1, H, W)
        resized = shadow_batched # differentiable_crop(shadow_batched, output_size=output_size)  # (1, 1, output_size, output_size)

        if shadowBoolean: 
            resized = differentiable_crop(shadow_batched, output_size=output_size)

        result.append(resized.squeeze(0))  # (1, H, W)

    return torch.stack(result, dim=0)  # (B, 1, H, W)

def differentiable_normalize_to_unit_box1(pts, temperature=10.0, eps=1e-6):
    """
    Normalize 2D points to fit inside [0, 1]^2 with preserved shape proportions.
    Fully differentiable using soft bounds and uniform scaling.
    Args:
        pts: (N, 2) torch.Tensor
    Returns:
        pts_normalized: (N, 2)
    """
    # Soft min/max per axis (differentiable)
    soft_min = torch.sum(pts * torch.softmax(-pts * temperature, dim=0), dim=0, keepdim=True)  # (1, 2)
    soft_max = torch.sum(pts * torch.softmax( pts * temperature, dim=0), dim=0, keepdim=True)  # (1, 2)

    extent = soft_max - soft_min  # (1, 2)

    # Uniform scaling: use the maximum extent across axes (approximate max)
    soft_extent = torch.sum(extent * torch.softmax(extent * temperature, dim=1), dim=1, keepdim=True)  # (1, 1)

    # Center and scale
    center = (soft_min + soft_max) / 2
    pts_centered = pts - center  # (N, 2)

    pts_scaled = pts_centered / (soft_extent + eps)  # uniform scale
    pts_normalized = pts_scaled + 0.5  # shift to [0, 1] range

    # Debug prints
    print("👉 Hard min:", pts_normalized.min(dim=0)[0])
    print("👉 Hard max:", pts_normalized.max(dim=0)[0])

    return pts_normalized
def differentiable_normalize_to_unit_box(pts, temperature=10.0, eps=1e-6):
    """
    Differentiably normalize all 2D points into [0, 1]^2 using soft min/max per axis.
    Args:
        pts: (N, 2) torch.Tensor
    Returns:
        pts_normalized: (N, 2)
    """
    # Soft min and max per axis (differentiable)
    soft_min = torch.sum(pts * torch.softmax(-pts * temperature, dim=0), dim=0, keepdim=True)  # (1, 2)
    soft_max = torch.sum(pts * torch.softmax( pts * temperature, dim=0), dim=0, keepdim=True)  # (1, 2)

    extent = soft_max - soft_min + eps  # (1, 2), prevent divide-by-zero

    # Normalize each axis to [0, 1]
    pts_normalized = (pts - soft_min) / extent

    pts_normalized = (pts - soft_min) / extent
    pts_normalized = torch.clamp(pts_normalized, 0.0, 1.0)

    return pts_normalized

def rotate_points_x_180(points):
    B, N, _ = points.shape
    rot_x_180 = torch.tensor([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=points.dtype, device=points.device)
    return torch.matmul(points, rot_x_180.T)

def rotate_mesh_y_45(mesh):
    """
    Rotates the mesh 45 degrees around the Y-axis, centered on the mesh.
    mesh: (B, P, 3)
    """
    angle_rad = torch.tensor(45.0 * torch.pi / 180.0, dtype=mesh.dtype, device=mesh.device)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    rot_y_45 = torch.tensor([
        [cos_a, 0, -sin_a],
        [0,     1, 0     ],
        [sin_a, 0, cos_a ]
    ], dtype=mesh.dtype, device=mesh.device)  # (3, 3)

    center = mesh.mean(dim=1, keepdim=True)
    mesh_centered = mesh - center
    rotated = torch.matmul(mesh_centered, rot_y_45.T)
    return rotated + center

def rotate_mesh_y_135(mesh):
    """
    Rotates the mesh 135 degrees around the Y-axis, centered on the mesh.
    mesh: (B, P, 3)
    """
    angle_rad = torch.tensor(135.0 * torch.pi / 180.0, dtype=mesh.dtype, device=mesh.device)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    rot_y_135 = torch.tensor([
        [cos_a, 0, -sin_a],
        [0,     1, 0     ],
        [sin_a, 0, cos_a ]
    ], dtype=mesh.dtype, device=mesh.device)  # (3, 3)

    center = mesh.mean(dim=1, keepdim=True)
    mesh_centered = mesh - center
    rotated = torch.matmul(mesh_centered, rot_y_135.T)
    return rotated + center

def soft_point_projection(vis, query_points, occupancy_pred, image_size=124, sigma=1.5, top_k=5000, axis='y', top_k_points=None, shadow_boolean=False):
    """
    Projects the top-K most occupied query points into a 2D differentiable shadow image.
    """
    if top_k_points is None:
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
    else: 
        topk_points = top_k_points
        topk_weights = None

    # return differentiable_point_projection(
    #     topk_points, image_size=image_size, sigma=sigma, output_size=image_size, weights=topk_weights, axis=axis, shadowBoolean=shadow_boolean
    # )

    image = differentiableProjection( 
        topk_points, image_size=image_size, sigma=sigma, output_size=image_size, weights=topk_weights, axis=axis
    ) # differentiable_point_projectionShadow(

    # vis.scatter(X=topk_points[0], Y=None, win='top-k-p', opts=dict(title="Top-K Points", markersize=2))

    return image

def soft_point_projectionShadow(vis, query_points, occupancy_pred, image_size=124, sigma=2.0, top_k=5000, axis='y', top_k_points=None, light_directions=None): # sigma=1.5
    """
    Projects the top-K most occupied query points into a 2D differentiable shadow image.
    """
    if top_k_points is None:
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
    else: 
        topk_points = top_k_points
        topk_weights = None

    if isinstance(topk_points, list):
        topk_points = torch.stack(topk_points, dim=0)
    topk_points_shadow = generate_shadow(vis, topk_points, light_directions)

    # vis.scatter(X=topk_points_shadow[0], Y=None, win='top-k', opts=dict(title="Top-K Points Shadow", markersize=2))

    image = differentiableProjection(
        topk_points_shadow, image_size=image_size, sigma=sigma, output_size=image_size, weights=topk_weights, axis=axis
    )# differentiable_point_projectionShadow(

    return image





