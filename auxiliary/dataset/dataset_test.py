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
    def __init__(self,
                 rootimg='/home/grammatikakis1/DATASET/ORTHOPHOTOS_N/', 
                 rootpc='/home/grammatikakis1/DATASET/TREES/', 
                 dsm_root='/home/grammatikakis1/DATASET/DSM/', 
                 train=True, npoints=2500, npoints_initial=2500,
                 use_dsm=True, SVR=True, extension='png',
                 max_files=4000, deciduous=False, num_trees=100, 
                 deciduous_only=False, fixed_ids=None):
        
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.npoints_initial = npoints_initial
        self.SVR = SVR
        self.extension = extension
        self.max_files = max_files
        self.use_dsm = use_dsm
        self.dsm_root = dsm_root
        self.num_trees = num_trees
        self.fixed_ids = fixed_ids
        
        # Initialize DSM files if needed
        self.dsm_files = []
        if self.use_dsm and self.dsm_root:
            print(f"Using DSM point clouds from {self.dsm_root}")
            self.dsm_files = sorted(glob.glob(os.path.join(self.dsm_root, "*.mat")))
            if len(self.dsm_files) == 0:
                raise FileNotFoundError(f"No DSM files found in {self.dsm_root}")

        # Get and filter filenames
        fns_img = sorted(os.listdir(self.rootimg))
        fns_pc = sorted(os.listdir(self.rootpc))
        fns = [fn for fn in fns_img if fn + '.mat' in fns_pc]

        # Extract index and split into ranges
        def extract_index(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else -1

        range_low = [fn for fn in fns if extract_index(fn) < 1000]
        range_high = [fn for fn in fns if 1000 < extract_index(fn) < 2000]
        range_high_con = [fn for fn in fns if extract_index(fn) > 2000]

        # Select files based on criteria
        if fixed_ids is not None:
            print(f"Using fixed IDs: {len(fixed_ids)} trees")
            # Ensure fixed_ids are strings (handle both string and array cases)
            if isinstance(fixed_ids, np.ndarray):
                fixed_ids = fixed_ids.tolist()
            if isinstance(fixed_ids[0], (np.str_, np.bytes_)):
                fixed_ids = [str(fid) for fid in fixed_ids]
            selected_fns = [fn for fn in fns if fn in fixed_ids]
            print(f"Found {len(selected_fns)} matching files from fixed IDs")
        else:
            if deciduous:
                # num_trees_high = num_trees // 2
                # selected_fns = (random.sample(range_low, min(num_trees_high, len(range_low))) +
                #               random.sample(range_high, min(num_trees, len(range_high))) +
                #               random.sample(range_high_con, min(num_trees_high, len(range_high_con))))
                num_trees_sub = num_trees // 3
                selected_fns = (random.sample(range_low, min(num_trees_sub, len(range_low))) +
                            random.sample(range_high, min(num_trees_sub, len(range_high))) +
                            random.sample(range_high_con, min(num_trees_sub, len(range_high_con))))
            elif deciduous_only:
                selected_fns = random.sample(range_high, min(num_trees, len(range_high)))
            else:
                selected_fns = random.sample(range_low, min(num_trees, len(range_low)))

        print(f"Total selected files: {len(selected_fns)}")

        # Build valid datapath
        valid_datapath = []
        files_index = {1, 2, 3, 4, 6, 8, 9}  # Pre-defined valid view indices
        
        for fn in selected_fns:
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
                    valid_datapath.append((img_path, pc_path, dsm_path))
                    
            except Exception as e:
                print(f"❌ Skipping {fn}: {e}")

        self.datapath = valid_datapath[:self.max_files] if self.max_files else valid_datapath
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
    
    def get_filenames(self):
        """Returns list of unique tree IDs used in this dataset."""
        return [os.path.basename(pc_path).split('.')[0] for _, pc_path, _ in self.datapath]

    def __getitem__(self, index):
        img_path, pc_path, dsm_path = self.datapath[index]
        
        # Load point cloud
        fp = sio.loadmat(pc_path)
        points = fp['vertices']
        indices = np.random.choice(points.shape[0], size=self.npoints)
        points = points[indices, :]

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
        
        if self.SVR:
            try:
                # Get rendering images
                render_path = os.path.join(img_path)
                if os.path.exists(render_path):
                    image_files = glob.glob(os.path.join(render_path, f"*.{self.extension}"))
                    files_index = {1, 2, 3, 4, 6, 8, 9}
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
        
        filename = os.path.basename(pc_path).split('.')[0]
        
        # Return values in the order expected by validation pipeline:
        # (orthophoto, gt_mesh, dsm_points, index_views, filenames, center_dsm, scale_dsm)
        return data, points, initial_vertices, index_view, filename, center_dsm, scale_dsm

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = TreeDataset(train=True, npoints=2500, SVR=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for i, data in enumerate(dataloader):
        img, points, init_verts, index_view, shadow_ortho, filename, center_dsm, scale_dsm = data
        print("Batch:", i, "Image shape:", img.shape, "Point Cloud shape:", points.shape)
        break