
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
                rootimg='/home/grammatikakis1/network-tree-gen/landmarks_austria/DATA_LANDMARKS/ORTHOPHOTOS/', 
                 dsm_root='/home/grammatikakis1/network-tree-gen/landmarks_austria/DATA_LANDMARKS/DSM/', 
                 species_file="/home/grammatikakis1/P2/TREES_DATASET/TREES/species_log_from_mtl.txt",
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
        
        # Load species data for classification
        self.species_to_label = {}
        self.species_list = []
        self.num_species = 0
        
        if species_file and os.path.exists(species_file):
            self.species_to_label = self.load_species_data(species_file)
        else:
            print("No species file provided, using default single-species labeling")
            # Set up default single species
            self.species_list = ['Unknown']
            self.num_species = 1
            self.species_to_label = {}  # Will be populated with default labels later
        
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
        
        # Filter to only include trees that have species information (only if species file was provided)
        if self.species_to_label and len(self.species_to_label) > 0:
            # Convert image filenames to tree names for species lookup
            selected_fns_with_species = []
            for fn in selected_fns:
                tree_name = fn.replace('.png', '')  # Convert tree_1.png to tree_1
                if tree_name in self.species_to_label:
                    selected_fns_with_species.append(fn)
            print(f"📊 Files with species information: {len(selected_fns_with_species)}/{len(selected_fns)}")
            selected_fns = selected_fns_with_species
        else:
            print(f"📊 Using all available files (no species filtering): {len(selected_fns)}")

        print('selected_fns:', selected_fns)

        # Final datapath with labels
        temp_datapath = []
        for fn in selected_fns:
            img_path = os.path.join(self.rootimg, fn)
            dsm_path = os.path.join(self.dsm_root, fn.replace('.png', '') + '.mat')
            tree_name = fn.replace('.png', '')  # Convert tree_1.png to tree_1
            
            # Get label from species data
            if self.species_to_label and len(self.species_to_label) > 0 and tree_name in self.species_to_label:
                # Use species-specific label
                label = self.species_to_label[tree_name]
                temp_datapath.append((img_path, dsm_path, label))
            elif len(self.species_to_label) == 0:
                # No species file provided, use default label 0
                label = 0
                temp_datapath.append((img_path, dsm_path, label))
                if len(temp_datapath) == 1:  # Print message only once
                    print("📋 No species file provided, using default labels")
            else:
                # Skip trees without species information
                print(f"⚠️ Skipping {tree_name}: No species information available")
                continue
            #     print(f"⚠️ Skipping {tree_name}: No species information available")
            #     continue
        
        self.datapath = temp_datapath

        # --------------------------------------------------------
        valid_datapath = []

        for img_path, dsm_path, label in self.datapath:
            if not os.path.exists(dsm_path):
                continue

            try:
                dsm_data = sio.loadmat(dsm_path)
                faces = dsm_data.get("faces", None)

                valid_datapath.append((img_path, dsm_path, label))
            except Exception as e:
                filename = os.path.basename(dsm_path)
                print(f"❌ Skipping {filename}: {e}")

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
        Normalize DSM points so that the minimum z-coordinate becomes 0 (ground level).
        
        Args:
            dsm_points: numpy array of shape (N, 3) where the dimensions are [x, y, z] and z is height
        
        Returns:
            normalized_dsm: numpy array of same shape with z-coordinates starting from 0
        """
        # Handle 2D array (N, 3)
        min_z = dsm_points[:, 2].min()  # Get minimum z coordinate (height)
        dsm_points[:, 2] -= min_z       # Translate so minimum z becomes 0
        
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

    def normalize_dsm_custom(self, points):
        """
        Custom DSM normalization:
        - Y-axis (height) scaled so maximum becomes 1
        - X and Z axes scaled to stay under 1
        - Ground level (min Y) becomes 0
        
        Args:
            points: numpy array of shape (N, 3) where columns are [x, y, z]
        
        Returns:
            normalized_points: Points with Y up to 1, X&Z under 1
            center: Center point for denormalization
            scale: Scale factor for denormalization
        """
        min_coord = points.min(axis=0)  # [min_x, min_y, min_z]
        max_coord = points.max(axis=0)  # [max_x, max_y, max_z]
        
        # Center point
        center = (max_coord + min_coord) / 2.0
        
        # Shift points so minimum Y becomes 0 (ground level)
        points_shifted = points.copy()
        points_shifted[:, 1] -= min_coord[1]  # Ground at Y=0
        
        # Center X and Z around 0
        points_shifted[:, 0] -= center[0]  # Center X
        points_shifted[:, 2] -= center[2]  # Center Z
        
        # Scale Y-axis so max height becomes 1
        y_range = max_coord[1] - min_coord[1]
        y_scale = 1.0 / (y_range + 1e-8)
        
        # Scale X and Z to stay under 1 (use smaller scale to keep them < 1)
        x_range = max_coord[0] - min_coord[0]
        z_range = max_coord[2] - min_coord[2]
        xz_scale = 0.8 / (max(x_range, z_range) + 1e-8)  # 0.8 to ensure they stay under 1
        
        # Apply different scales to different axes
        normalized = points_shifted.copy()
        normalized[:, 0] *= xz_scale  # X axis
        normalized[:, 1] *= y_scale   # Y axis (height)
        normalized[:, 2] *= xz_scale  # Z axis
        
        # Shift to [0, 1] range
        normalized[:, 0] += 0.5  # X centered at 0.5
        normalized[:, 2] += 0.5  # Z centered at 0.5
        # Y already starts at 0, goes up to 1
        
        # Store scale information for denormalization
        scale = max(y_range, max(x_range, z_range))  # Use original max range for denormalization
        
        print(f"[DEBUG] DSM normalization - Y range: [0, {normalized[:, 1].max():.3f}], "
              f"X range: [{normalized[:, 0].min():.3f}, {normalized[:, 0].max():.3f}], "
              f"Z range: [{normalized[:, 2].min():.3f}, {normalized[:, 2].max():.3f}]")
        
        return normalized, center, scale

    def __getitem__(self, index):
        img_path, dsm_path, label = self.datapath[index]
        print('img_path:', img_path)
        print('dsm_path:', dsm_path)
        print('label:', label)
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
        volume_scale = 2000  # Increased from 500 - Higher value = more points
        min_points = 2000    # Increased minimum points
        max_points = 15000   # Increased maximum points

        if self.use_dsm and self.dsm_root:
            try:
                dsm_data = sio.loadmat(dsm_path)

                if 'vertices' in dsm_data:
                    dsm_points = dsm_data['vertices']

                    # Keep original for debugging - make a copy before any processing
                    original_dsm = dsm_points.copy()

                    dsm_points, center_dsm, scale_dsm = self.normalize_to_unit_cube(dsm_points)

                    # Apply custom DSM normalization: Y-axis up to 1, X,Z under 1
                 #   dsm_points, center_dsm, scale_dsm = self.normalize_dsm_custom(dsm_points)

                    
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


                    if len(dsm_points) >= self.npoints_initial:
                        indices_dsm = np.random.choice(dsm_points.shape[0], size=self.npoints_initial, replace=False)
                        initial_vertices = dsm_points[indices_dsm, :]
                        original_dsm = original_dsm[indices_dsm, :]
                    else:
                        # If DSM has fewer points than needed, sample with replacement
                        indices_dsm = np.random.choice(dsm_points.shape[0], size=self.npoints_initial, replace=True)
                        initial_vertices = dsm_points[indices_dsm, :]
                        original_dsm = original_dsm[indices_dsm, :]
                    
                    # # If we want more points than available, use all available points
                    # if dynamic_npoints_dsm == available_points:
                    #     indices_dsm = np.arange(available_points)
                    #     initial_vertices = dsm_points
                    #     original_dsm = original_dsm  # Keep all original points
                    # else:
                    #     indices_dsm = np.random.choice(available_points, size=dynamic_npoints_dsm, replace=False)
                    #     initial_vertices = dsm_points[indices_dsm, :]
                    #     # Also subsample original_dsm to match
                    #     original_dsm = original_dsm[indices_dsm, :]
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

        return data, initial_vertices, index_view, filename, center_dsm, scale_dsm, original_dsm, label

    def __len__(self):
        return len(self.datapath)
    
    def get_species_info(self):
        """Return species information for the dataset."""
        if hasattr(self, 'species_list') and self.species_list:
            return {
                'num_species': self.num_species,
                'species_list': self.species_list,
                'species_to_label': {species: i for i, species in enumerate(self.species_list)}
            }
        else:
            return {
                'num_species': 1,
                'species_list': ['Unknown'],
                'species_to_label': {'Unknown': 0}
            }

    def get_label_to_species(self, label):
        """Convert a label back to species name."""
        if hasattr(self, 'species_list') and self.species_list and label < len(self.species_list):
            return self.species_list[label]
        else:
            return f"Unknown_{label}"

    def load_species_data(self, species_file_path):
        """Load species data from TXT file and create species-to-label mapping with unique labels per species."""
        species_to_label = {}
        species_list = []  # Keep track of unique species for labeling
        
        if os.path.exists(species_file_path):
            try:
                with open(species_file_path, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line or line.lower().startswith('tree_id'):
                            continue  # Skip empty lines and header
                        
                        # Handle both comma and tab separated values
                        if ',' in line:
                            parts = line.split(',')
                        elif '\t' in line:
                            parts = line.split('\t')
                        else:
                            # Try splitting by whitespace as fallback
                            parts = line.split()
                        
                        if len(parts) >= 2:
                            tree_id, species = parts[0].strip(), parts[1].strip()
                            
                            # Convert 5-digit tree IDs to 4-digit by removing leading zero
                            # e.g., "tree_02000" -> "tree_2000"
                            if len(tree_id) > 8 and tree_id.startswith('tree_0') and tree_id[5:].isdigit():
                                # Extract the numeric part, convert to int to remove leading zeros, then back to string
                                numeric_part = tree_id[5:]  # Get "02000"
                                normalized_numeric = str(int(numeric_part))  # Convert to "2000"
                                tree_id = f"tree_{normalized_numeric}"  # Result: "tree_2000"
                                print(f'🔄 Normalized tree ID: {parts[0].strip()} -> {tree_id}')

                            print(f'🌳 Loading: {tree_id} -> {species}')
                            
                            # Create unique label for each species
                            if species not in species_list:
                                species_list.append(species)
                            
                            species_label = species_list.index(species)
                            species_to_label[tree_id] = species_label
                
                print(f"✅ Loaded species data for {len(species_to_label)} trees")
                print(f"📊 Found {len(species_list)} unique species:")
                
                # Print species distribution
                species_counts = {}
                for tree_id, label in species_to_label.items():
                    species_name = species_list[label]
                    species_counts[species_name] = species_counts.get(species_name, 0) + 1
                
                # Sort by count for better readability
                sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (species_name, count) in enumerate(sorted_species):
                    print(f"   {i}: {species_name} - {count} trees")
                
                # Store species list for later reference
                self.species_list = species_list
                self.num_species = len(species_list)
                
            except Exception as e:
                print(f"❌ Error loading species file {species_file_path}: {e}")
                species_to_label = {}
                self.species_list = []
                self.num_species = 0
        else:
            print(f"⚠️  Species file not found: {species_file_path}")
            self.species_list = []
            self.num_species = 0
            
        return species_to_label
