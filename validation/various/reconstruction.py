import numpy as np
import torch
import open3d as o3d

def ball_pivot_mesh(points, radii=[0.005, 0.01, 0.02]):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh
def estimate_alpha(points, multiplier=1.5):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    dists = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(dists)
    return multiplier * avg_dist
def alpha_shape_mesh(points, alpha=None):
    alpha = estimate_alpha(points, multiplier=2.5) if alpha is None else alpha
    print('alpha:', alpha)
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=alpha * 1.5, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    return mesh
def ball_pivoting(points, multiplier=2.5):
    """
    Fast BPA mesh reconstruction with cleanup using a fixed radius.
    
    Args:
        points: torch.Tensor or np.ndarray of shape (N, 3)
        multiplier: float — scales the average NN distance to define radius
    
    Returns:
        mesh: cleaned TriangleMesh
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    dists = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(dists)
    radius = multiplier * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 1.5])
    )

    # Clean
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    return mesh
def poisson_mesh(points, depth=9, trim=True):
    """
    Detailed surface reconstruction using Poisson Surface Reconstruction.
    
    Args:
        points: (N, 3) torch.Tensor or np.ndarray
        depth: Octree depth — controls resolution. 8–10 is usually good.
        trim: Whether to remove low-density regions (optional cleanup)

    Returns:
        mesh: o3d.geometry.TriangleMesh
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals (critical for Poisson!)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    if trim:
        # Remove vertices in low-density regions
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.02)
        vertices_to_keep = densities > density_threshold
        mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
        mesh.remove_unreferenced_vertices()

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    return mesh

def create_dense_grid(dsm_points, resolution=128):
    """
    Creates a dense 3D grid (res³ points) within the AABB of each batched DSM input.

    Args:
        dsm_points: Tensor of shape (B, N, 3)
        resolution: int (number of grid points per axis)

    Returns:
        grid_points: Tensor of shape (B, resolution³, 3)
    """
    B = dsm_points.shape[0]
    num_query_points = resolution ** 3

    min_xyz = dsm_points.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
    max_xyz = dsm_points.max(dim=1, keepdim=True)[0]  # (B, 1, 3)

    # Random samples in [0,1]^3 then scale to bbox
    rand_unit = torch.rand((B, num_query_points, 3), device=dsm_points.device)
    query_points = rand_unit * (max_xyz - min_xyz) + min_xyz

    return query_points

def estimate_ball_radius(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    return 3 * avg_dist
