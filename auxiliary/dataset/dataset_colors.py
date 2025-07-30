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
    def __init__(self, rootimg='/home/grammatikakis1/TREES_DATASET/ORTHOPHOTOS/', # /usr/people/agrammat/Desktop/
                 rootpc='/home/grammatikakis1/TREES_DATASET/TREES/', 
                 dsm_root='/home/grammatikakis1/TREES_DATASET/DSM/', # DATASET/DSM/', 
                 species_file="/home/grammatikakis1/TREES_DATASET/TREES/species_log_from_mtl.txt",
                 train=True, npoints=2500, npoints_initial=2500,
                 use_dsm=True, extension='png',
                 max_files=4000, train_test_split=0.85,
                 num_trees=100):
        
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
        fns = [fn for fn in fns_img if fn + '.mat' in fns_pc]

        # Limit to num_trees if specified
        if num_trees is not None and num_trees > 0:
            fns = fns[:num_trees]

        # Extract index and split into ranges
        def extract_index(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else -1

        print(f"Total selected files: {len(fns)}")

        # Build valid datapath with labels
        valid_datapath = []
        files_index = {1, 2, 3, 4, 5, 6, 7} #  8, 9}  # Pre-defined valid view indices
        
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
        
    def __getitem__(self, index):
        img_path, pc_path, dsm_path, label = self.datapath[index]
        
        # Load point cloud and colors
        fp = sio.loadmat(pc_path)
        points = fp['vertices']
        colors = fp['colors']
        
        # Sample points and corresponding colors
        indices = np.random.choice(points.shape[0], size=self.npoints)
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
                    dsm_points, center_dsm, scale_dsm = self.normalize_to_unit_cube(dsm_points)
                    # points, center_dsm, scale_dsm = self.normalize_to_unit_cube(points)

                    # 1st change
                    points = (points - center_dsm) / scale_dsm + 0.5
                    
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
        return data, points, initial_vertices, index_view, shadow_ortho, label, point_colors

    def __len__(self):
        return len(self.datapath)

    def load_species_data(self, species_file_path):
        """Load species data from TXT file and create species-to-label mapping."""
        # Define coniferous and deciduous species
        coniferous_species = {'Pine', 'Spruce', 'Fir', 'Cedar', 'Juniper', 'Cypress', 'Larch', 'Yew'}
        deciduous_species = {'Oak', 'Maple', 'Birch', 'Ash', 'Elm', 'Beech', 'Alder', 'Plane', 
                           'Linden', 'Hornbeam', 'Willow', 'Poplar', 'Cherry', 'Walnut', 'Chestnut','Aspen'}
        
        species_to_label = {}
        
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
                            
                            # Classify species
                            if species in coniferous_species:
                                species_to_label[tree_id] = 0  # Coniferous
                            elif species in deciduous_species:
                                species_to_label[tree_id] = 1  # Deciduous
                            else:
                                # Default classification for unknown species
                                print(f"⚠️  Unknown species '{species}' for {tree_id}, defaulting to coniferous (0)")
                                species_to_label[tree_id] = 0
                                
                print(f"✅ Loaded species data for {len(species_to_label)} trees")
                
                # Print species distribution
                coniferous_count = sum(1 for label in species_to_label.values() if label == 0)
                deciduous_count = sum(1 for label in species_to_label.values() if label == 1)
                print(f"   Coniferous: {coniferous_count}, Deciduous: {deciduous_count}")
                
            except Exception as e:
                print(f"❌ Error loading species file {species_file_path}: {e}")
                species_to_label = {}
        else:
            print(f"⚠️  Species file not found: {species_file_path}")
            
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