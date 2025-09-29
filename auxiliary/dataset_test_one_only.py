import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TreeDatasetOneOnly(Dataset):
    """
    Loads a single tree sample (DSM, ortho, tree) for a given tree_id.
    """
    def __init__(self, dataset_root, tree_id, num_points=2500, dsm_filename=None):
        # num_points: number of points to sample from DSM
        self.num_points = num_points
        # If tree_id is just a number, convert to proper format
        if tree_id.isdigit():
            num = int(tree_id)
            tree_id_str = f"tree_{num:04d}" if num < 1000 else f"tree_{num}"
            folder_name = f"tree_{num}" if num >= 1000 else f"tree_{num:04d}"
            # Always use tree_0XXXX.mat for DSM if > 1000
            if num >= 1000:
                dsm_filename = f"tree_{num:05d}.mat"
            else:
                dsm_filename = f"tree_{num:04d}.mat"
            tree_id = tree_id_str

        self.dataset_root = dataset_root
        self.tree_id = tree_id
        # DSM and tree from .mat files, ortho from .png
        if dsm_filename is not None:
            self.dsm_path = os.path.join(dataset_root, f"DSM/{dsm_filename}")
        else:
            self.dsm_path = os.path.join(dataset_root, f"DSM/{tree_id}.mat")
        self.ortho_path = os.path.join(dataset_root, f"ORTHOPHOTOS/{tree_id}/rendering/view_005.png")
        if num >= 1000:
            self.ortho_path = os.path.join(dataset_root, f"ORTHOPHOTOS/tree_{num:05d}/rendering/view_005.png")
        else:
            self.ortho_path = os.path.join(dataset_root, f"ORTHOPHOTOS/tree_{num:04d}/rendering/view_005.png")
        

        # Optionally, add query_points_path if needed
        self.query_points_path = os.path.join(dataset_root, f"QUERY_POINTS/{tree_id}.npy")
        # Check existence
        for p in [self.dsm_path, self.ortho_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file for tree_id {tree_id}: {p}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Load DSM, ortho, tree, query_points
        # DSM: tree_id.mat from DSM/
        # Use the resolved DSM path
        dsm = self.load_mat(self.dsm_path)
        # Ortho: rendering/view_005.png from ORTHOPHOTOS/tree_id/rendering/
        ortho_img_path = os.path.join(self.dataset_root, f"ORTHOPHOTOS/{self.tree_id}/rendering/view_005.png")
        
        num = int(self.tree_id.split('_')[-1])
        if num >= 1000:
            ortho_img_path = os.path.join(self.dataset_root, f"ORTHOPHOTOS/tree_{num:05d}/rendering/view_005.png")

        from PIL import Image
        import torchvision.transforms as transforms
        img = Image.open(ortho_img_path).convert('RGB')
        rgb_transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ortho = rgb_transforms(img)
        # Tree: tree_id.mat from TREES/
        tree_mat_path = os.path.join(self.dataset_root, f"TREES/{self.tree_id}.mat")
        num = int(self.tree_id.split('_')[-1])
        if num >= 1000:
            tree_mat_path = os.path.join(self.dataset_root, f"TREES/tree_{num:05d}.mat")
        tree = self.load_mat(tree_mat_path)
        # Query points
        query_points = self.load_query_points(self.query_points_path)
        # After loading dsm, subsample to num_points
        if dsm.shape[0] > self.num_points:
            idxs = np.random.choice(dsm.shape[0], self.num_points, replace=False)
            dsm = dsm[idxs, :]
        elif dsm.shape[0] < self.num_points:
            idxs = np.random.choice(dsm.shape[0], self.num_points, replace=True)
            dsm = dsm[idxs, :]

        if tree.shape[0] > self.num_points:
            idxs = np.random.choice(tree.shape[0], self.num_points, replace=False)
            tree = tree[idxs, :]
        elif tree.shape[0] < self.num_points:
            idxs = np.random.choice(tree.shape[0], self.num_points, replace=True)
            tree = tree[idxs, :]

        # Ensure batch dimension for dsm and ortho
        dsm = dsm.unsqueeze(0)  # [1, num_points, 3]
        ortho = ortho.unsqueeze(0)  # [1, 3, 224, 224]
        return {
            'tree_id': self.tree_id,
            'dsm': dsm,
            'ortho': ortho,
            'tree': tree,
            'query_points': query_points
        }

    def load_tif(self, path):
        # Load DSM, ortho, tree, query_points
        dsm = self.load_mat(self.dsm_path)
        ortho = self.load_mat(self.ortho_path)
        tree = self.load_ply(self.tree_path)
        query_points = self.load_query_points(self.query_points_path)
        return {
            'tree_id': self.tree_id,
            'dsm': dsm,
            'ortho': ortho,
            'tree': tree,
            'query_points': query_points
        }

    def normalize_to_unit_cube(self, points):
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord) / 2.0
        points_centered = points - center
        extent = (max_coord - min_coord).max()
        scale = extent + 1e-6
        normalized = points_centered / scale + 0.5
        return normalized

    def load_mat(self, path):
        # Load DSM or ortho from .mat file as in TreeDataset
        import scipy.io as sio
        mat = sio.loadmat(path)
        if 'vertices' in mat:
            points = mat['vertices']
            normalized = self.normalize_to_unit_cube(points)
            return torch.from_numpy(normalized).float()
        else:
            raise ValueError(f"No 'vertices' found in {path}")

    def load_ply(self, path):
        # Dummy loader, replace with your actual PLY loader
        # Example: return np.loadtxt(path, skiprows=10)
        return torch.zeros((1000, 3))

    def load_query_points(self, path):
        # Dummy loader, replace with your actual query points loader
        if os.path.exists(path):
            return torch.from_numpy(np.load(path)).float()
        else:
            return torch.zeros((10000, 3))
