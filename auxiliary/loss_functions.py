import torch
import torch.nn.functional as F

import sys
sys.path.append('./extension/')
import dist_chamfer as ext
chamfer_distance = ext.chamferDist()

# ===========================
# Loss Functions
# ===========================
def reconstruction_loss(pred, gt):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    dist1, dist2, _, _ = chamfer_distance(pred, gt)
    return torch.mean(dist1) + torch.mean(dist2)

def l2_shadow_loss(pred, target):
    return F.mse_loss(pred, target)

def shadow_consistency_loss(real_shadow, generated_shadow):
    return torch.nn.functional.mse_loss(real_shadow, generated_shadow)

def occupancy_loss(pred_occupancy, gt_occupancy, weights=None):
    if weights is None:
        weights = torch.ones_like(gt_occupancy)
    loss = F.binary_cross_entropy(pred_occupancy, gt_occupancy, weight=weights)
    # loss = F.binary_cross_entropy_with_logits(pred_occupancy, gt_occupancy, weight=weights)
    return loss

def occupancy_loss_with_logits(pred_occupancy, gt_occupancy): # , weights):
    # if weights is None:
    #     weights = torch.ones_like(gt_occupancy)
    criterion = torch.nn.BCEWithLogitsLoss() # weight=weights)
    # pred_occupancy = pred_occupancy * 5  # sharpen sigmoid curve
    return criterion(pred_occupancy, gt_occupancy)

def suppress_fp_loss(logits, gt_occupancy, fp_weight=2.0, fn_weight=1.5):
    """
    Differentiable loss emphasizing suppression of false positives and false negatives.
    """
    probs = torch.sigmoid(logits)
    gt = gt_occupancy

    # Soft FP and FN masks
    FP_mask = (1 - gt) * probs
    FN_mask = gt * (1 - probs)

    # Weights: only penalize FP and FN
    weights = FP_mask * fp_weight #+ FN_mask * fn_weight

    loss = F.binary_cross_entropy_with_logits(logits, gt, weight=weights.detach(), reduction='none')

    # Logging
    soft_fp = FP_mask.sum().item()
    soft_fn = FN_mask.sum().item()
    soft_tp = (gt * probs).sum().item()

    print(f"🟦 Soft FP: {soft_fp:.1f}  🔴 Soft FN: {soft_fn:.1f}  ✅ Soft TP: {soft_tp:.1f}")

    return loss.mean()

def suppress_fp_loss(logits, gt_occupancy, fp_weight=3.0, fn_weight=1.5, tp_weight=1.0):
    """
    A loss that penalizes False Positives (FP) and False Negatives (FN) more than True Positives (TP).
    """
    probs = torch.sigmoid(logits)
    gt = gt_occupancy

    # Soft masks for each class
    FP_mask = (1 - gt) * probs           # High when gt=0 and pred is close to 1
    FN_mask = gt * (1 - probs)           # High when gt=1 and pred is close to 0
    TP_mask = gt * probs                 # High when both pred and gt are 1

    # Base BCE loss (element-wise)
    bce_loss = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')

    # Create dynamic weighting map
    weight_map = FP_mask * fp_weight + FN_mask * fn_weight + TP_mask * tp_weight

    # Apply weights
    weighted_loss = bce_loss * weight_map.detach()  # detach to not backprop through weights

    # Debugging and stats (optional)
    soft_fp = FP_mask.sum().item()
    soft_fn = FN_mask.sum().item()
    soft_tp = TP_mask.sum().item()
    print(f"🟦 Soft FP: {soft_fp:.1f}  🔴 Soft FN: {soft_fn:.1f}  ✅ Soft TP: {soft_tp:.1f}")

    return weighted_loss.mean()

def soft_iou_loss1(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) if pred.max() > 1 or pred.min() < 0 else pred
    target = torch.sigmoid(target) if target.max() > 1 or target.min() < 0 else target

    intersection = torch.sum(pred * target, dim=(1,2,3))
    union = torch.sum(pred, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()

def soft_iou_loss(pred, target, eps=1e-6):
    spatial_dims = tuple(range(1, pred.ndim))  # all dims except batch
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=spatial_dims)
    union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()

import torch
import torch.nn.functional as F

def sobel_kernel():
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # (2, 1, 3, 3)
    return kernel.cuda()

def edge_loss(pred_mask, gt_mask):
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
    elif pred_mask.shape[0] == 1 and pred_mask.ndim == 4:
        pred_mask = pred_mask.permute(1, 0, 2, 3)  # (1, B, H, W) → (B, 1, H, W)

    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    elif gt_mask.shape[0] == 1 and gt_mask.ndim == 4:
        gt_mask = gt_mask.permute(1, 0, 2, 3)
    
    # pred_mask and gt_mask should be of shape (B, 1, H, W)
    pred_edges = F.conv2d(pred_mask, sobel_kernel(), padding=1)
    gt_edges = F.conv2d(gt_mask, sobel_kernel(), padding=1)
    return F.l1_loss(pred_edges, gt_edges)

def sharpness_loss(occupancy_pred, query_points):
    """
    Computes sharpness loss by penalizing smooth transitions in predicted occupancy.
    Args:
        occupancy_pred: (B, N, 1) occupancy probabilities from model
        query_points:   (B, N, 3) input query coordinates
    Returns:
        Scalar sharpness loss
    """
    B, N, _ = query_points.shape
    occupancy_pred = occupancy_pred.view(B, N)

    # Approximate gradient by finite differences between neighboring points
    # (Just pick a random nearby point in the batch for now)
    idx = torch.roll(torch.arange(N), shifts=1).to(query_points.device)
    pred_shifted = occupancy_pred[:, idx]
    dist = torch.norm(query_points - query_points[:, idx, :], dim=-1) + 1e-8  # prevent div by 0

    grad = torch.abs(occupancy_pred - pred_shifted) / dist  # gradient magnitude
    return -grad.mean()  # NEGATIVE to encourage sharper transitions

def shadow_loss(pred_shadow, generated_shadow):
    return torch.nn.functional.mse_loss(pred_shadow, generated_shadow)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal loss with logits.
    Args:
        logits: raw outputs from model (B, N, 1)
        targets: ground truth occupancy (B, N, 1)
        alpha: weighting factor for class imbalance
        gamma: focusing parameter
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probas = torch.sigmoid(logits)

    pt = targets * probas + (1 - targets) * (1 - probas)  # pt = p if t==1 else (1 - p)
    loss = (alpha * (1 - pt) ** gamma) * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def eikonal_loss(gradients):
    return ((gradients.norm(2, dim=2) - 1) ** 2).mean()

import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np

def knn(x, k):
    """
    x: (B, N, 3) tensor of 3D points
    returns: (B, N, k) indices of k-nearest neighbors
    """
    B, N, _ = x.shape
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, N, 1)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def compute_normals_differentiable(points, k=10):
    """
    points: (B, N, 3)
    returns: (B, N, 3) approximate normals
    """
    B, N, _ = points.shape
    idx = knn(points, k=k)  # (B, N, k)

    idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)

    points_flat = points.view(B * N, -1)
    neighbor_points = points_flat[idx].view(B, N, k, 3)

    # Center neighbors
    centroid = points.unsqueeze(2)  # (B, N, 1, 3)
    neighbors = neighbor_points - centroid  # (B, N, k, 3)

    # Covariance matrix
    cov = torch.matmul(neighbors.transpose(3, 2), neighbors)  # (B, N, 3, 3)
    _, _, V = torch.linalg.svd(cov)  # SVD: U, S, V

    normals = V[:, :, :, -1]  # last column = smallest eigenvalue's vector
    return F.normalize(normals, dim=-1)  # (B, N, 3)

def normal_consistency_loss(pred_points, gt_points):
    pred_normals = compute_normals_differentiable(pred_points)
    gt_normals = compute_normals_differentiable(gt_points)

    loss = 1.0 - F.cosine_similarity(pred_normals, gt_normals, dim=-1)
    return loss.mean()

import lpips
def local_lpips_loss(loss_fn, img_pred, img_gt, patch_size=32, num_patches=10):
    B, C, H, W = img_pred.shape
    total_loss = 0.0

    for _ in range(num_patches):
        top = torch.randint(0, H - patch_size, (1,)).item()
        left = torch.randint(0, W - patch_size, (1,)).item()

        patch_pred = img_pred[:, :, top:top+patch_size, left:left+patch_size]
        patch_gt = img_gt[:, :, top:top+patch_size, left:left+patch_size]

        total_loss += loss_fn(patch_pred, patch_gt).mean()

    return total_loss / num_patches

def curvature_aware_loss(occ_pred, query_points, k=10):
    """
    Approximates curvature as variance in occupancy between a point and its k nearest neighbors.

    query_points: (B, N, 3)
    occ_pred:     (B, N, 1)
    """
    B, N, _ = query_points.shape

    # Normalize query points for numerical stability
    q = query_points / (query_points.max(dim=1, keepdim=True)[0] + 1e-8)  # (B, N, 3)

    # Compute pairwise distances (B, N, N)
    dists = torch.cdist(q, q)  # shape: (B, N, N)

    # Get indices of k nearest neighbors (excluding self)
    knn_inds = dists.topk(k=k+1, largest=False).indices[:, :, 1:]  # (B, N, k)

    # Gather occupancy of neighbors
    # occ_pred: (B, N, 1) → (B, N)
    occ_flat = occ_pred.squeeze(-1)  # (B, N)

    # Expand for gathering
    knn_inds_expanded = knn_inds.unsqueeze(-1).expand(-1, -1, -1, 1)  # (B, N, k, 1)
    occ_neighbors = torch.gather(occ_flat.unsqueeze(1).expand(-1, N, -1), 2, knn_inds)  # (B, N, k)

    occ_center = occ_flat.unsqueeze(-1)  # (B, N, 1)
    occ_diff = occ_neighbors - occ_center  # (B, N, k)
    occ_var = torch.var(occ_diff, dim=-1)  # (B, N)

    print("Occ var mean:", occ_var.mean().item())
    print("Occ var std:", occ_var.std().item())

    # High variance = detail. We penalize low curvature with exp(-variance)
    penalty = torch.exp(-5 * occ_var)  # near 1 if variance is ~0
    loss = penalty.mean()
    return loss

def compute_shadow_loss(pred_shadow_img, gt_shadow_mask, loss_type="bce"):
    if loss_type == "bce":
        return F.binary_cross_entropy(pred_shadow_img, gt_shadow_mask)
    elif loss_type == "l1":
        return F.l1_loss(pred_shadow_img, gt_shadow_mask)
    elif loss_type == "mse":
        return F.mse_loss(pred_shadow_img, gt_shadow_mask)
    else:
        raise NotImplementedError(f"Loss {loss_type} not supported.")
    
from torchvision.models import vgg16
from torchvision.transforms import Normalize

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(pretrained=True).features.to(device).eval()
        self.vgg = vgg
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Freeze VGG weights
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Layers to extract features from (relu1_2, relu2_2, relu3_3, relu4_3)
        self.layer_ids = [3, 8, 15, 22]
        
    def forward(self, pred, target):
        # Normalize inputs to match VGG training stats
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Extract features at multiple layers
        loss = 0
        for i in range(max(self.layer_ids)+1):
            pred = self.vgg[i](pred)
            target = self.vgg[i](target)
            if i in self.layer_ids:
                loss += F.l1_loss(pred, target)
                
        return loss / len(self.layer_ids)