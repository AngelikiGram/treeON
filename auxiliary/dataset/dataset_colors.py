from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import scipy.io as sio
import glob
from utils import *
import random
import re

def crop_to_white_region(image_path):
    """Opens a grayscale image and crops to the non-black region."""
    img = Image.open(image_path).convert('L')
    np_img = np.array(img)
    white_mask = np_img < 10
    coords = np.argwhere(white_mask)
    if coords.shape[0] > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))
    return img

def normalize_to_origin(pc):
    """Normalize point cloud to origin with unit scale."""
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    if scale > 0:
        pc = pc / scale
    min_vals = pc.min(axis=0, keepdims=True)
    pc = pc - min_vals
    overall_range = (pc.max(axis=0) - pc.min(axis=0)).max() + 1e-8
    return pc / overall_range

class TreeDataset(data.Dataset):
    def __init__(self, rootimg='/home/grammatikakis1/TREES_DATASET/ORTHOPHOTOS/', # /home/grammatikakis1/
                 rootpc='/home/grammatikakis1/TREES_DATASET/TREES/', 
                 dsm_root='/home/grammatikakis1/TREES_DATASET/DSM-higherRes/', # DATASET/DSM/', 
                 species_file="/home/grammatikakis1/TREES_DATASET/TREES/species_log_from_mtl.txt",
                 train=True, npoints=2500, npoints_initial=2500,
                 use_dsm=True, extension='png',
                 max_files=4000, train_test_split=0.85,
                 num_trees=100, many_trees=False):
        
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.npoints_initial = npoints_initial
        self.SVR = True
        self.extension = extension
        self.max_files = max_files
        self.train_test_split = train_test_split
        self.use_dsm = use_dsm
        self.dsm_root = dsm_root
        self.many_trees = many_trees

        self.num_trees = num_trees  # Limit on number of trees to use, None for all available
        
        # Load species data for classification
        self.species_to_label = {}
        if species_file and os.path.exists(species_file):
            self.species_to_label = self.load_species_data(species_file)
        else:
            print("No species file provided, using default labeling")
        
        # Initialize DSM files if needed
        self.dsm_files = []
        if self.use_dsm and self.dsm_root:
            print(f"Using DSM point clouds from {self.dsm_root}")
            self.dsm_files = sorted(glob.glob(os.path.join(self.dsm_root, "*.mat")))
            if len(self.dsm_files) == 0:
                raise FileNotFoundError(f"No DSM files found in {self.dsm_root}")

        # Get and filter filenames efficiently
        fns_img = sorted(os.listdir(self.rootimg))
        fns_pc = sorted(os.listdir(self.rootpc))
        # Select only files present in both image and point cloud folders
        all_fns = [fn for fn in fns_img if fn + '.mat' in fns_pc]

        print(f"📊 Total files found: {len(all_fns)}")
        
        # Filter to only include trees that have species information
        if self.species_to_label:
            all_fns_with_species = [fn for fn in all_fns if fn in self.species_to_label]
            print(f"📊 Files with species information: {len(all_fns_with_species)}/{len(all_fns)}")
            all_fns = all_fns_with_species

        # Extract index function for filtering
        def extract_index(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else -1

        # # Handle tree selection based on many_trees parameter and num_trees limit
        # if num_trees is not None and num_trees > 0:
        #     # Split files into ranges based on tree ID
        #     range1 = [fn for fn in all_fns if 1 <= extract_index(fn) <= 1800]  # 1-1800
        #     range2 = [fn for fn in all_fns if 1801 <= extract_index(fn) <= 3000]  # 1801-3000
        #     range3 = [fn for fn in all_fns if extract_index(fn) > 10000]  # >10000
            
        #     print(f"📊 Available trees by range:")
        #     print(f"   Range 1 (1-1800): {len(range1)} trees")
        #     print(f"   Range 2 (1801-3000): {len(range2)} trees")
        #     print(f"   Range 3 (>10000): {len(range3)} trees")
            
        #     if self.many_trees:
        #         # many_trees=True: 50% from 1801-3000, 27% from 1-1800, 23% from >10000
        #         target_range2 = int(num_trees * 0.40)  # 50%
        #         target_range1 = int(num_trees * 0.37)  # 27%
        #         target_range3 = num_trees - target_range2 - target_range1  # 23% (remaining)
                
        #         print(f"📊 many_trees=True - Target distribution:")
        #         print(f"   Range 1 (1-1800): {target_range1} trees (37%)")
        #         print(f"   Range 2 (1801-3000): {target_range2} trees (40%)")
        #         print(f"   Range 3 (>10000): {target_range3} trees (23%)")
        #     else:
        #         # many_trees=False: 60% from 1801-3000, 40% from 1-1800, 0% from >10000
        #         target_range2 = int(num_trees * 0.60)  # 60%
        #         target_range1 = num_trees - target_range2  # 40% (remaining)
        #         target_range3 = 0  # 0%
                
        #         print(f"📊 many_trees=False - Target distribution:")
        #         print(f"   Range 1 (1-1800): {target_range1} trees (40%)")
        #         print(f"   Range 2 (1801-3000): {target_range2} trees (60%)")
        #         print(f"   Range 3 (>10000): {target_range3} trees (0%)")
            
        #     # Sample from each range
        #     selected_fns = []
            
        #     # Sample from range 1 (1-1800)
        #     if target_range1 > 0 and len(range1) > 0:
        #         actual_range1 = min(target_range1, len(range1))
        #         random.shuffle(range1)
        #         selected_fns.extend(range1[:actual_range1])
        #         print(f"   ✅ Selected {actual_range1} trees from range 1")
            
        #     # Sample from range 2 (1801-3000)  
        #     if target_range2 > 0 and len(range2) > 0:
        #         actual_range2 = min(target_range2, len(range2))
        #         random.shuffle(range2)
        #         selected_fns.extend(range2[:actual_range2])
        #         print(f"   ✅ Selected {actual_range2} trees from range 2")
            
        #     # Sample from range 3 (>10000) - only if many_trees=True
        #     if target_range3 > 0 and len(range3) > 0 and self.many_trees:
        #         actual_range3 = min(target_range3, len(range3))
        #         random.shuffle(range3)
        #         selected_fns.extend(range3[:actual_range3])
        #         print(f"   ✅ Selected {actual_range3} trees from range 3")
            
        #     # If we don't have enough trees, fill from any available range
        #     if len(selected_fns) < num_trees:
        #         remaining_needed = num_trees - len(selected_fns)
        #         available_remaining = [fn for fn in all_fns if fn not in selected_fns]
        #         if self.many_trees:
        #             # Include all ranges
        #             pass  # available_remaining already includes all
        #         else:
        #             # Exclude range 3 (>10000)
        #             available_remaining = [fn for fn in available_remaining if extract_index(fn) <= 10000]
                
        #         if len(available_remaining) > 0:
        #             actual_additional = min(remaining_needed, len(available_remaining))
        #             random.shuffle(available_remaining)
        #             selected_fns.extend(available_remaining[:actual_additional])
        #             print(f"   ➕ Added {actual_additional} additional trees to reach target")
            
        #     fns = selected_fns
        #     print(f"📊 Final selection: {len(fns)} trees")
        # else:
        #     # Use all available trees when num_trees is not specified
        #     if self.many_trees:
        #         fns = all_fns
        #         print(f"📊 many_trees=True: Using all {len(fns)} available trees")
        #     else:
        #         # Exclude range 3 (>10000) when many_trees=False
        #         fns = [fn for fn in all_fns if extract_index(fn) <= 10000]
        #         print(f"📊 many_trees=False: Using all {len(fns)} available trees (ID <= 10000)")

        # Select just num_trees randomly from all_fns
        if self.num_trees is not None and self.num_trees > 0:
            random.shuffle(all_fns)
            fns = all_fns[:self.num_trees]
            print(f"📊 Randomly selected {len(fns)} trees out of {len(all_fns)} available.")
        else:
            fns = all_fns
            print(f"📊 Using all {len(fns)} available trees.")


        print(f"Total selected files: {len(fns)}")

        # Build valid datapath with labels
        valid_datapath = []
        files_index = {1, 2, 3, 4, 5, 6, 7} #  8, 9}  # Pre-defined valid view indices
        
        filtered_count = 0  # Count of files filtered out due to Y-axis criteria
        
        for fn in fns:
            img_path = os.path.join(self.rootimg, fn, "rendering")
            pc_path = os.path.join(self.rootpc, fn + '.mat')
            dsm_path = os.path.join(self.dsm_root, fn + '.mat')
            
            if not os.path.exists(dsm_path):
                continue

            try:
                # Quick validation of DSM file
                dsm_data = sio.loadmat(dsm_path)
                if 'vertices' not in dsm_data:
                    continue
                
                # Validate main point cloud file and Y-axis coverage
                if not os.path.exists(pc_path):
                    continue
                    
                pc_data = sio.loadmat(pc_path)
                if 'vertices' not in pc_data:
                    continue
                
                # Check Y-axis coverage - normalize points and check if any points are in [0.2, 0.5] range
                pc_vertices = pc_data['vertices']
                # Simple normalization for Y-axis check
                min_coord = pc_vertices.min(axis=0)
                max_coord = pc_vertices.max(axis=0)
                center = (max_coord + min_coord) / 2.0
                points_centered = pc_vertices - center
                extent = (max_coord - min_coord).max()
                scale = extent + 1e-6
                pc_normalized = points_centered / scale + 0.5
                
                y_coords = pc_normalized[:, 1]
                points_in_y_range = np.any((y_coords >= 0.4) & (y_coords <= 0.6))
                
                if not points_in_y_range:
                    print(f"⚠️  Skipping {fn}: No points in Y-axis range [0.4, 0.6]")
                    filtered_count += 1
                    continue
                    
                # Check if image directory has valid views
                image_files = glob.glob(os.path.join(img_path, f"*.{self.extension}"))
                valid_views = [f for f in image_files 
                             if int(os.path.basename(f).split('_')[-1].split('.')[0]) in files_index]
                
                if valid_views:  # Only add if has valid views
                    # Assign labels based on species data
                    tree_id = fn  # filename should match tree_id format
                    if self.species_to_label and tree_id in self.species_to_label:
                        label = self.species_to_label[tree_id]
                    else:
                        # Default label if no species data available
                        # Extract number from filename for simple heuristic
                        idx = extract_index(fn)
                        label = 0 if idx % 2 == 0 else 1  # Simple alternating default
                        if not self.species_to_label:
                            print(f"Using default label {label} for {fn}")
                    
                    valid_datapath.append((img_path, pc_path, dsm_path, label))
                    
            except Exception as e:
                print(f"❌ Skipping {fn}: {e}")

        self.datapath = valid_datapath[:self.max_files] if self.max_files else valid_datapath

        # Train/test split
        random.shuffle(self.datapath)
        split_idx = int(len(self.datapath) * self.train_test_split)
        self.train_data = self.datapath[:split_idx]
        self.test_data = self.datapath[split_idx:]
        self.datapath = self.train_data if train else self.test_data

        print(f'Valid datapath: {len(self.datapath)}')
        print(f'Filtered out due to Y-axis criteria: {filtered_count} files')
        if filtered_count > 0:
            print(f'Filtering removed {filtered_count}/{len(fns)} files ({filtered_count/len(fns)*100:.1f}%)')

        # Initialize transforms once
        self.rgb_transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
    def spatial_uniform_sampling(self, points, num_samples):
        """
        Sample points spatially uniformly from all sides of the mesh.
        This ensures better coverage of the 3D space compared to random sampling.
        Always returns exactly num_samples indices.
        """
        if len(points) <= num_samples:
            # If we have fewer points than needed, sample with replacement to get exact count
            return np.random.choice(len(points), size=num_samples, replace=True)
        
        # Divide 3D space into a grid and sample from each cell
        # This ensures spatial uniformity across all axes
        
        # Find bounding box
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # Handle edge case where all points are identical
        if np.allclose(min_coords, max_coords):
            return np.random.choice(len(points), size=num_samples, replace=False)
        
        # Determine grid resolution based on number of samples
        # Aim for roughly cube root of num_samples per dimension
        grid_res = max(2, int(np.cbrt(num_samples / 4)))  # Conservative estimate
        
        # Create 3D grid with small epsilon to handle boundary cases
        eps = 1e-6
        x_bins = np.linspace(min_coords[0] - eps, max_coords[0] + eps, grid_res + 1)
        y_bins = np.linspace(min_coords[1] - eps, max_coords[1] + eps, grid_res + 1)
        z_bins = np.linspace(min_coords[2] - eps, max_coords[2] + eps, grid_res + 1)
        
        # Assign each point to a grid cell
        x_indices = np.digitize(points[:, 0], x_bins) - 1
        y_indices = np.digitize(points[:, 1], y_bins) - 1
        z_indices = np.digitize(points[:, 2], z_bins) - 1
        
        # Clamp indices to valid range
        x_indices = np.clip(x_indices, 0, grid_res - 1)
        y_indices = np.clip(y_indices, 0, grid_res - 1)
        z_indices = np.clip(z_indices, 0, grid_res - 1)
        
        # Create unique cell identifiers
        cell_ids = x_indices * (grid_res * grid_res) + y_indices * grid_res + z_indices
        
        # Group points by cell
        unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
        
        # Calculate how many points to sample from each cell
        total_cells = len(unique_cells)
        base_samples_per_cell = max(1, num_samples // total_cells)
        extra_samples = num_samples % total_cells
        
        sampled_indices = []
        
        # Sample from each cell
        for i, cell_id in enumerate(unique_cells):
            cell_mask = (cell_ids == cell_id)
            cell_point_indices = np.where(cell_mask)[0]
            
            # Determine how many samples from this cell
            samples_from_cell = base_samples_per_cell
            if i < extra_samples:
                samples_from_cell += 1
                
            # Sample points from this cell
            if samples_from_cell >= len(cell_point_indices):
                # Take all points from this cell (with replacement if needed)
                if len(cell_point_indices) == 0:
                    continue
                selected = np.random.choice(cell_point_indices, size=samples_from_cell, replace=True)
            else:
                # Randomly sample from this cell without replacement
                selected = np.random.choice(cell_point_indices, size=samples_from_cell, replace=False)
            
            sampled_indices.extend(selected)
        
        # Convert to numpy array
        sampled_indices = np.array(sampled_indices)
        
        # Ensure we have exactly the right number of samples
        if len(sampled_indices) < num_samples:
            # Need more points - add random samples from entire point cloud
            remaining_needed = num_samples - len(sampled_indices)
            additional = np.random.choice(len(points), size=remaining_needed, replace=True)
            sampled_indices = np.concatenate([sampled_indices, additional])
        elif len(sampled_indices) > num_samples:
            # Too many points - randomly subsample to exact count
            sampled_indices = np.random.choice(sampled_indices, size=num_samples, replace=False)
        
        # Final validation
        assert len(sampled_indices) == num_samples, f"Expected {num_samples} samples, got {len(sampled_indices)}"
        
        return sampled_indices

    def __getitem__(self, index):
        img_path, pc_path, dsm_path, label = self.datapath[index]
        
        # Load point cloud and colors
        fp = sio.loadmat(pc_path)
        points = fp['vertices']
        colors = fp['colors']
        
        # First normalize points to check Y-axis coverage
        points_normalized_temp, _, _ = self.normalize_to_unit_cube(points)
        
        # Check if there are any points in the Y-axis range [0.2, 0.5]
        y_coords = points_normalized_temp[:, 1]  # Y-axis coordinates
        points_in_range = np.any((y_coords >= 0.2) & (y_coords <= 0.5))
        
        if not points_in_range:
            # Skip this sample - no points in the required Y range
            raise ValueError(f"No points found in Y-axis range [0.2, 0.5] for sample {index}")
        
        # Sample points spatially uniformly from all sides of the mesh
        # indices = np.random.choice(points.shape[0], size=self.npoints)
        indices = self.spatial_uniform_sampling(points, self.npoints)
        points = points[indices, :]
        point_colors = colors[indices, :]

        # Initialize variables with correct size
        initial_vertices = np.random.rand(self.npoints_initial, 3).astype(np.float32)  # Ensure correct size
        center_dsm = np.zeros(3)
        scale_dsm = 1.0

        # Load DSM if available
        if self.use_dsm and self.dsm_root:
            try:
                dsm_data = sio.loadmat(dsm_path)
                if 'vertices' in dsm_data:
                    dsm_points = dsm_data['vertices']
                    # Normalize DSM points and get transformation parameters
                    dsm_points, center_dsm, scale_dsm = self.normalize_to_unit_cube(dsm_points)
                    points, _, _ = self.normalize_to_unit_cube(points)
                    
                    # Apply the same transformation to the main points to align them with DSM
                    # This ensures both point clouds are in the same coordinate system
                    # points = (points - center_dsm) / scale_dsm + 0.5
                    
                    # Sample DSM points to ensure consistent size
                    if len(dsm_points) >= self.npoints_initial:
                        indices_dsm = np.random.choice(dsm_points.shape[0], size=self.npoints_initial, replace=False)
                        initial_vertices = dsm_points[indices_dsm, :]
                    else:
                        # If DSM has fewer points than needed, sample with replacement
                        indices_dsm = np.random.choice(dsm_points.shape[0], size=self.npoints_initial, replace=True)
                        initial_vertices = dsm_points[indices_dsm, :]
                        
            except Exception as e:
                print(f"Warning: Failed to load DSM {dsm_path}: {e}")
                # Ensure initial_vertices has correct size even on failure
                initial_vertices = np.random.rand(self.npoints_initial, 3).astype(np.float32)

        # Load image if SVR mode
        data = torch.zeros((3, 224, 224))  # Default fallback
        index_view = 0
        
        try:
            # Get rendering images
            render_path = os.path.join(img_path)
            if os.path.exists(render_path):
                image_files = glob.glob(os.path.join(render_path, f"*.{self.extension}"))
                files_index = {1, 2, 3, 4, 5, 6, 7}
                valid_files = [f for f in image_files 
                                if int(f.split('_')[-1].split('.')[0]) in files_index]
                
                if valid_files:
                    img_file = random.choice(valid_files)
                    image = Image.open(img_file).convert('RGB')
                    data = self.rgb_transforms(image)
                    index_view = int(os.path.basename(img_file).split('_')[-1].split('.')[0])
                    
        except Exception as e:
            print(f"Warning: Failed to load image from {img_path}: {e}")

        # Final validation to ensure consistent tensor sizes
        assert points.shape == (self.npoints, 3), f"Points shape mismatch: {points.shape} != ({self.npoints}, 3)"
        assert initial_vertices.shape == (self.npoints_initial, 3), f"Initial vertices shape mismatch: {initial_vertices.shape} != ({self.npoints_initial}, 3)"
        assert point_colors.shape == (self.npoints, 3), f"Point colors shape mismatch: {point_colors.shape} != ({self.npoints}, 3)"
        
        shadow_ortho = data  # Simplified - same as main data
        return data, points, initial_vertices, index_view, shadow_ortho, label, point_colors, None

    def __len__(self):
        return len(self.datapath)
    def load_species_data(self, species_file_path):
        """Load species data from TXT/CSV file and create species-to-label mapping with unique labels per species."""
        species_to_label = {}
        species_list = []  # Keep track of unique species for labeling
        import csv
        if os.path.exists(species_file_path):
            try:
                with open(species_file_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        tree_id, species = row[0].strip(), row[1].strip()
                        # Normalize tree_id to match file format (e.g., tree_0001)
                        if tree_id.startswith('tree_') and len(tree_id) == 9:
                            pass  # already correct
                        elif tree_id.startswith('tree_') and tree_id[5:].isdigit():
                            tree_id = f"tree_{int(tree_id[5:]):04d}"
                        else:
                            # fallback: try to extract number and format
                            m = re.search(r'(\d+)', tree_id)
                            if m:
                                tree_id = f"tree_{int(m.group(1)):04d}"
                        if species not in species_list:
                            species_list.append(species)
                        species_label = species_list.index(species)
                        species_to_label[tree_id] = species_label
                print(f"✅ Loaded species data for {len(species_to_label)} trees")
                print(f"📊 Found {len(species_list)} unique species:")
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


if __name__ == '__main__':
    # Example usage with species file
    species_file_path = "d:/TREES_DATASET/TREES/species_log_from_mtl.txt"
    dataset = TreeDataset(train=True, npoints=2500, SVR=True, species_file=species_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for i, data in enumerate(dataloader):
        img, points, init_verts, index_view, shadow_ortho, label, point_colors = data
        print("Batch:", i, "Image shape:", img.shape, "Point Cloud shape:", points.shape, 
              "Colors shape:", point_colors.shape, "Labels:", label.numpy())
        break