
from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image, ImageOps
import scipy.io as sio
import time
import glob
from utils import *
from scipy.spatial import ConvexHull

import random
import re

import torchvision.transforms as T

def normalize_to_origin(pc):
    # Center at mean
    pc = pc - np.mean(pc, axis=0)
    # Uniform scale to [-1, 1]
    scale = np.max(np.linalg.norm(pc, axis=1))
    if scale > 0:
        pc = pc / scale

    # Translate so lowest point is at origin
    min_vals = pc.min(axis=0, keepdims=True)
    pc = pc - min_vals  # Ensures minimum is at 0
    # Scale again so longest axis becomes 1
    overall_range = (pc.max(axis=0) - pc.min(axis=0)).max() + 1e-8
    pc = pc / overall_range
    return pc
    

def get_boundary_vertices(faces):
    """
    Returns a boolean array where True indicates that the vertex is part of a boundary edge.
    """
    from collections import defaultdict
    from scipy.spatial import ConvexHull

    edge_count = defaultdict(int)

    for face in faces:
        # Sort vertices in edge to avoid directional duplicates
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Collect boundary vertex indices
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)

    return np.array(sorted(boundary_vertices))

class TreeDataset(data.Dataset):
    def __init__(self,
                rootimg='/home/grammatikakis1/p2-tree-gen/landmarks_austria/ORTHOPHOTOS/', 
                 dsm_root='/home/grammatikakis1/p2-tree-gen/landmarks_austria/DSM/', 
                 train=True, npoints=2500, npoints_initial=2500, normal=False,
                 use_dsm=True,
                 SVR=False, extension='png',
                 max_files=4000,  
                 train_test_split=0.85,
                 deciduous=False, 
                 num_trees=100, 
                 deciduous_only=False): # 85):
        
        self.train = train
        self.rootimg = rootimg
        self.npoints = npoints
        self.npoints_initial = npoints # npoints_initial
        self.SVR = SVR
        self.extension = extension
        self.max_files = max_files
        self.train_test_split = train_test_split
        self.use_dsm = use_dsm
        self.dsm_root = dsm_root
        
        if self.use_dsm and self.dsm_root:
            print(f"Using DSM point clouds from {self.dsm_root}")
            self.dsm_files = sorted(glob.glob(os.path.join(self.dsm_root, "*.mat")))
            if len(self.dsm_files) == 0:
                raise FileNotFoundError(f"No DSM files found in {self.dsm_root}")
        else:
            self.dsm_files = []

        fns_img = sorted(os.listdir(self.rootimg))
        fns_dsm = sorted(os.listdir(self.dsm_root))

        # --------------------------------------------------------
        # ------------- Filter based on tree index -------------
        # --------------------------------------------------------

        def extract_index(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else -1

        # Match image and PC files by name
        selected_fns = [fn for fn in fns_img if fn.replace('.png', '') + '.mat' in fns_dsm]

        print('selected_fns:', selected_fns)

        # Final datapath
        self.datapath = [(os.path.join(self.rootimg, fn), os.path.join(self.dsm_root, fn + '.mat')) for fn in selected_fns]

        # --------------------------------------------------------
        valid_datapath = []

        for fn in selected_fns:
            img_path = os.path.join(self.rootimg, fn)
            dsm_path = os.path.join(self.dsm_root, fn.replace('.png', '') + '.mat')
            
            if not os.path.exists(dsm_path):
                continue

            try:
                dsm_data = sio.loadmat(dsm_path)
                faces = dsm_data.get("faces", None)

                valid_datapath.append((img_path, dsm_path))
            except Exception as e:
                print(f"❌ Skipping {fn}: {e}")

        self.datapath = valid_datapath

        print('Total files:', len(self.datapath))

        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.rgb_transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        self.gray_transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.ToTensor(),  # Automatically gives (1, H, W)
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
        ])

    # def normalize_to_unit_cube(self, points):
    #     """
    #     Normalize 3D points to [0, 1]^3 using the same scale across all axes.
    #     Returns:
    #         normalized_points: Tensor in [0, 1]^3
    #         center: (3,) tensor for inverse transform
    #         scale: scalar for inverse transform
    #     """
    #     min_coord = points.min(axis=0)
    #     max_coord = points.max(axis=0)
    #     center = (max_coord + min_coord) / 2.0

    #     # Translate to origin
    #     points_centered = points - center

    #     # Use max extent across all axes to preserve proportions
    #     extent = (max_coord - min_coord).max()
    #     scale = extent + 1e-6  # Avoid divide-by-zero

    #     # Normalize to [-0.5, 0.5] → then shift to [0, 1]
    #     normalized = points_centered / scale + 0.5

    #     return normalized, center, scale

    def normalize_to_unit_cube(self, points):
        """Normalize 3D points to [0, 1]^3 using same scale across all axes."""
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord) / 2.0
        points_centered = points - center
        extent = (max_coord - min_coord).max()
        scale = extent + 1e-6
        normalized = points_centered / scale + 0.5
        return normalized, center, scale
    
    def normalize_dsm_to_ground_zero(self, dsm_points):
        """
        Normalize DSM points so that the minimum y-coordinate becomes 0 (ground level).
        
        Args:
            dsm_points: numpy array of shape (N, 3) where the dimensions are [x, y, z] and y is height
        
        Returns:
            normalized_dsm: numpy array of same shape with y-coordinates starting from 0
        """
        # Handle 2D array (N, 3)
        min_y = dsm_points[:, 1].min()  # Get minimum y coordinate (height)
        dsm_points[:, 1] -= min_y       # Translate so minimum y becomes 0
        
        return dsm_points
    
    def normalize_to_unit_cubeA(self, points):
        """
        Normalize 3D points to [0, 1]^3 using the same scale across all axes.
        Centers the points at (0, 0) in x and z axes.
        Returns:
            normalized_points: Tensor in [0, 1]^3
            center: (3,) tensor for inverse transform
            scale: scalar for inverse transform
        """
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        
        # Calculate center for x, z axes (center at 0, 0)
        # For y-axis, use min_coord to keep ground level
        center = (max_coord + min_coord) / 2.0
        center[1] = min_coord[1]  # Keep y-axis at ground level
        
        # Center x and z at origin, but keep y at ground level
        points_centered = points.copy()
        points_centered[:, 0] -= center[0]  # Center x at 0
        points_centered[:, 2] -= center[2]  # Center z at 0
        points_centered[:, 1] -= min_coord[1]  # Move ground to y=0

        # Use max extent across all axes to preserve proportions
        extent = (max_coord - min_coord).max()
        scale = extent + 1e-6  # Avoid divide-by-zero

        # Normalize to [-0.5, 0.5] → then shift to [0, 1]
        normalized = points_centered / scale + 0.5

        # Update center to reflect the centering we applied
        center[0] = (max_coord[0] + min_coord[0]) / 2.0  # x center
        center[2] = (max_coord[2] + min_coord[2]) / 2.0  # z center
        center[1] = min_coord[1]  # y ground level

        return normalized, center, scale

    # def normalize_to_unit_cube(self, points):
    #     min_coord = points.min(axis=0)
    #     max_coord = points.max(axis=0)

    #     scale = max_coord - min_coord + 1e-6
    #     normalized = (points - min_coord) / scale

    #     return normalized, min_coord, scale

    def normalize_to_unit_cube1(self, pc):
        """
        Uniformly scale and translate point cloud so that:
        - the highest Y point becomes 1
        - the lowest point in all axes becomes 0
        - shape proportions in X, Y, Z are preserved

        Args:
            pc (np.ndarray): (N, 3)

        Returns:
            pc_normalized (np.ndarray): (N, 3) normalized point cloud
        """
        min_vals = pc.min(axis=0, keepdims=True)           # (1, 3)
        pc_shifted = pc - min_vals                         # shift lowest point to 0

        ymax = pc_shifted[:, 1].max()                      # highest Y after shift
        scale = 1.0 / (ymax + 1e-8)                        # uniform scale factor so Ymax = 1

        pc_normalized = pc_shifted * scale                 # scale uniformly
        return pc_normalized

    def __getitem__(self, index):
        img_path, dsm_path = self.datapath[index]
        print('img_path:', img_path)
        print('dsm_path:', dsm_path)
## 
        # Direct matching orthophoto: same filename as DSM but .png extension
        img_path = os.path.join(self.rootimg, os.path.basename(dsm_path).replace('.mat', '.png'))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Orthophoto not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        data = self.rgb_transforms(image)
        index_view = 0  # You can keep index_view if needed, set to 0 by default here
##

        # ## VOLUME
        # # Compute the bounding box volume
        # min_bounds = points.min(axis=0)
        # max_bounds = points.max(axis=0)
        # volume = np.prod(max_bounds - min_bounds)
        # # Normalize volume to define point count
        # volume = max(volume, 1e-6)  # Prevent divide-by-zero
        volume_scale = 2000  # Increased from 500 - Higher value = more points
        min_points = 2000    # Increased minimum points
        max_points = 15000   # Increased maximum points
        # # Dynamic number of points based on volume
        # dynamic_npoints = int(volume * volume_scale)
        # dynamic_npoints = np.clip(dynamic_npoints, 500, 5000)  # Clamp to [500, 5000]
        # # Sample dynamic number of points
        # indices = np.random.choice(points.shape[0], size=dynamic_npoints)
        # points = points[indices, :]

        if self.use_dsm and self.dsm_root:
            try:
                dsm_data = sio.loadmat(dsm_path)

                if 'vertices' in dsm_data:
                    dsm_points = dsm_data['vertices']

                    # Keep original for debugging - make a copy before any processing
                    original_dsm = dsm_points.copy()

                    dsm_points = self.normalize_dsm_to_ground_zero(dsm_points)
                    # min_coord = dsm_points.min(axis=0)
                    # max_coord = dsm_points.max(axis=0)
                    # center_dsm = (max_coord + min_coord) / 2.0
                    # scale_dsm = (max_coord - min_coord).max() + 1e-6
                    # dsm_points = (dsm_points - min_coord) / (max_coord - min_coord + 1e-6)

                    dsm_points, center_dsm, scale_dsm = self.normalize_to_unit_cube(dsm_points)

                    
                    initial_vertices = dsm_points 


                    ## VOLUME (Convex Hull)
                    # Compute convex hull volume for DSM points
                    try:
                        hull = ConvexHull(dsm_points)
                        dsm_volume = hull.volume
                    except Exception as e:
                        print(f"Convex hull volume computation failed: {e}")
                        # Fallback to bounding box volume if convex hull fails
                        min_dsm = dsm_points.min(axis=0)
                        max_dsm = dsm_points.max(axis=0)
                        dsm_volume = np.prod(max_dsm - min_dsm)

                    dynamic_npoints_dsm = int(dsm_volume * volume_scale)
                    dynamic_npoints_dsm = np.clip(dynamic_npoints_dsm, min_points, max_points)
                    
                    # Ensure we don't try to sample more points than available
                    available_points = dsm_points.shape[0]
                    dynamic_npoints_dsm = min(dynamic_npoints_dsm, available_points)
                    
                    # If we want more points than available, use all available points
                    if dynamic_npoints_dsm == available_points:
                        indices_dsm = np.arange(available_points)
                        initial_vertices = dsm_points
                        original_dsm = original_dsm  # Keep all original points
                    else:
                        indices_dsm = np.random.choice(available_points, size=dynamic_npoints_dsm, replace=False)
                        initial_vertices = dsm_points[indices_dsm, :]
                        # Also subsample original_dsm to match
                        original_dsm = original_dsm[indices_dsm, :]
                else:
                    raise ValueError(f"'vertices' key not found in DSM file: {dsm_path}")

            except Exception as e:
                print(f"Warning: Failed to load DSM {dsm_path}: {e}")
                # Ensure initial_vertices has correct size even on failure
                initial_vertices = np.random.rand(self.npoints_initial, 3).astype(np.float32)

        # # Sample initial vertices
        # if self.use_dsm and self.dsm_root:
        #     # Ensure we don't sample more points than available
        #     num_available = initial_vertices.shape[0]
        #     num_to_sample = min(self.npoints_initial, num_available)
            
        #     indices = np.random.choice(num_available, size=num_to_sample, replace=False)
        #     initial_vertices = initial_vertices[indices, :]

        #     # Make sure original_dsm has the same number of points before indexing
        #     if original_dsm.shape[0] == num_available:
        #         original_dsm = original_dsm[indices, :]
        #     else:
        #         # If sizes don't match, sample from original_dsm separately
        #         orig_num_available = original_dsm.shape[0]
        #         orig_num_to_sample = min(num_to_sample, orig_num_available)
        #         orig_indices = np.random.choice(orig_num_available, size=orig_num_to_sample, replace=False)
        #         original_dsm = original_dsm[orig_indices, :]
                
        #         # If we need more points, pad with zeros or repeat
        #         if orig_num_to_sample < num_to_sample:
        #             padding_needed = num_to_sample - orig_num_to_sample
        #             # Repeat the last few points to match the size
        #             repeat_indices = np.random.choice(orig_num_to_sample, size=padding_needed, replace=True)
        #             padding = original_dsm[repeat_indices, :]
        #             original_dsm = np.vstack([original_dsm, padding])

        filename = os.path.basename(img_path).split('.')[0]

        return data, initial_vertices, index_view, filename, center_dsm, scale_dsm, original_dsm

    def __len__(self):
        return len(self.datapath)
