from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
import os
import json
import datetime
import visdom
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms

sys.path.append('./extension/')
import dist_chamfer as ext
chamfer_distance = ext.chamferDist()

sys.path.append('./auxiliary/')
from utils import AverageValueMeter
from shadow_gen import *    
from loss_functions import *
import torch.nn.functional as F
from load_model import load_checkpoint, load_curves
from visualize import visualize
from query_points_gen import *
from occupancy_compute import *

sys.path.append('./auxiliary/dataset/')
from dataset_categories import TreeDataset # dataset_colors

from kornia.color import rgb_to_lab

sys.path.append('./auxiliary/models/')

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# ===========================
# Argument Parser
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16) # 16 # 32) # 32) # 64)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--nepoch', type=int, default=650) # 450) # 700
parser.add_argument('--num_points', type=int, default=4000) # 1500) # 4000)
parser.add_argument('--super_points', type=int, default=4000) # 1500) # 4000)
parser.add_argument('--env', type=str, default="p2")
parser.add_argument('--lr', type=float, default=1e-2) # 2) # 1)
parser.add_argument('--manualSeed', type=int, default=6185)
parser.add_argument('--model_previous_training', type=str, default='')
parser.add_argument('--port', type=int, default=8092)
parser.add_argument('--image_size', type=int, default=124) 
parser.add_argument('--num_query', type=int, default=10000)
parser.add_argument('--top_k', type=int, default=2500)
parser.add_argument('--deciduous', type=bool, default=False) 
parser.add_argument('--num_trees', type=int, default=100)
parser.add_argument('--thres', type=int, default=20)
parser.add_argument('--deciduous_only', type=bool, default=False)
parser.add_argument('--variable', type=str, default='1') # 0) # 1) # 2)
parser.add_argument('--model', type=str, default='1')
parser.add_argument('--dsm_convex_hull', type=bool, default=False) # True) # False)

parser.add_argument('--bce', type=bool, default=False) 
parser.add_argument('--silhouettes', type=bool, default=False) 
parser.add_argument('--shadow', type=bool, default=False) 
parser.add_argument('--classes_loss', type=bool, default=False) # True) # False)
parser.add_argument('--colors', type=bool, default=False) 

parser.add_argument('--save_images', type=bool, default=False, help='Save visualization images to disk')

parser.add_argument('--top_k_gt_occupancy', type=bool, default=False, help='Use top-k occupancy for GT points')

parser.add_argument('--spruces', type=bool, default=False, help='Use spruces dataset')
parser.add_argument('--top_silh', type=bool, default=False, help='Use top silhouette loss')

parser.add_argument('--many_trees', type=bool, default=False, help='Use many trees dataset')
opt = parser.parse_args()
print(opt)

def collate_fn(batch):
    imgs, points, initial_vertices, index_views, shadow_orthos, labels, colors, filenames = zip(*batch)

    imgs = torch.stack([torch.tensor(img, dtype=torch.float32) if isinstance(img, np.ndarray) else img for img in imgs])
    points = torch.stack([torch.tensor(p, dtype=torch.float32) if isinstance(p, np.ndarray) else p for p in points])
    initial_vertices = torch.stack([torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v for v in initial_vertices])
    shadow_ortho = torch.stack([torch.tensor(img, dtype=torch.float32) if isinstance(img, np.ndarray) else img for img in shadow_orthos])
    index_views = torch.tensor(index_views)
    labels = torch.tensor(labels, dtype=torch.long)
    # colors = torch.stack([torch.tensor(c, dtype=torch.float32) if isinstance(c, np.ndarray) else c for c in colors])

    colors = torch.stack([torch.tensor(c, dtype=torch.float32) if isinstance(c, np.ndarray) else c for c in colors])
    # Normalize GT colors per-sample to [0,1]
    max_vals = colors.view(colors.size(0), -1, 3).amax(dim=1, keepdim=True)  # [B,1,3]
    max_vals[max_vals == 0] = 1.0
    colors = colors / max_vals

    filenames = [f if isinstance(f, str) else f.decode('utf-8') for f in filenames]  # Ensure filenames are strings

    return imgs, points, initial_vertices, index_views, shadow_ortho, labels, colors, filenames

# ----------------------

def vis_images(vis, gt_shadow_img1, title='GT'):
    shadow_gt = F.interpolate(gt_shadow_img1, size=(256, 256), mode='bilinear', align_corners=False)
    shadow_gt_np = shadow_gt[0].detach().cpu().numpy()
    shadow_gt_np = (shadow_gt_np - shadow_gt_np.min()) / (shadow_gt_np.max() - shadow_gt_np.min() + 1e-8)
    vis.image(
                        shadow_gt_np,
                        win=title,
                        opts=dict(title=title, caption=title)
                    )

def vis_gt_colored_points(vis, gt_points, matched_colors_list, device, title="GT Colored Points", window="gt_colored_points"):
    """
    Visualizes ground truth colored points in a point cloud format.
    """
    gt_points_shape = gt_points[0].shape
    # Pad both points and colors together
    gt_points_tensor = pad_or_trim(
        [p.to(dtype=torch.float32, device=device) for p in gt_points],
        target_k=gt_points_shape[0]
    )
    gt_colors_tensor = pad_or_trim(
        [torch.tensor(c, dtype=torch.float32, device=device) if isinstance(c, np.ndarray) else c.to(dtype=torch.float32, device=device) for c in matched_colors_list],
        target_k=gt_points_shape[0]
    )
    vis_point_cloud(
        vis,
        gt_points_tensor,           # [1, K, 3]
        color=gt_colors_tensor,     # [1, K, 3]
        title=f"GT Colored Points",
        window="gt_colored_points"
    )

def vis_point_cloud(vis, pc, color=None, title="Point Cloud", window=None):
    """
    pc: Tensor of shape (N, 3)
    color: Tensor or ndarray of shape (N, 3), with float in [0, 1] or int in [0, 255]
    """
    pc = pc[0].squeeze(0).detach().cpu() if pc.dim() == 3 else pc.detach().cpu()
    pc = pc[:, [1, 0, 2]] # 0, 2, 1]]

    if color is not None:
        color_aligned = color[0].squeeze(0).detach().cpu() if isinstance(color, torch.Tensor) and color.dim() == 3 else color.detach().cpu() if isinstance(color, torch.Tensor) else color

        markercolor = None
        if color is not None:
            if isinstance(color, torch.Tensor):
                color = color[0].squeeze(0).detach().cpu() if color.dim() == 3 else color.detach().cpu()
                markercolor = (color * 255).clamp(0, 255).to(torch.uint8).numpy()
            elif isinstance(color, np.ndarray):
                markercolor = (color[0] * 255).clip(0, 255).astype(np.uint8) if color.ndim == 3 else (color * 255).clip(0, 255).astype(np.uint8)

        opts = dict(
            title=title,
            markersize=6
        )
        if markercolor is not None:
            opts["markercolor"] = markercolor

        return vis.scatter(X=pc, opts=opts, win=window)
    else:
        print(f"Warning: color shape {color_aligned.shape} does not match pc shape {pc.shape}, skipping lowest color print.")

def histogram_match_gt_colors(gt_mesh, gt_colors, orthophoto):
    """
    Samples color from orthophoto at GT mesh positions (x,z), 
    but first histogram-matches orthophoto to GT color distribution.
    
    Args:
        gt_mesh: [B, P, 3] GT mesh coordinates
        gt_colors: [B, P, 3] original GT RGB in [0, 1]
        orthophoto: [B, 3, H, W] orthophoto RGB in [0, 1]
    
    Returns:
        matched_sampled_colors: [B, P, 3] histogram-matched sampled colors
    """
    B, P, _ = gt_mesh.shape
    _, _, H, W = orthophoto.shape

    coords = gt_mesh[:, :, [0, 2]]     # [B, P, 2] (x, z)
    coords_norm = coords * 2 - 1       # normalized for grid_sample

    grid = coords_norm.unsqueeze(2).unsqueeze(2)  # [B, P, 1, 1, 2]
    grid = grid.permute(0, 1, 3, 2, 4).reshape(B, P, 1, 2)  # [B, P, 1, 2]

    matched_colors = []

    for b in range(B):
        ortho = orthophoto[b].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        gt_col = gt_colors[b].detach().cpu().numpy()                     # [P, 3]

        # Match orthophoto histogram to GT colors
        ortho_matched = match_histograms(ortho.reshape(-1, 3), gt_col, channel_axis=1)
        ortho_matched = ortho_matched.reshape(H, W, 3).transpose(2, 0, 1)  # [3, H, W]

        # Back to tensor
        ortho_matched_tensor = torch.tensor(ortho_matched, device=gt_mesh.device).unsqueeze(0)  # [1, 3, H, W]

        # Sample
        grid_b = grid[b].unsqueeze(0)  # [1, P, 1, 2]
        sampled = F.grid_sample(ortho_matched_tensor, grid_b, mode='bilinear', align_corners=True)  # [1, 3, P, 1]
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [P, 3]

        sampled = sampled.to(gt_colors.device)
        blended = 0.7 * gt_colors[b] + 0.3 * sampled  # GT dominates
        matched_colors.append(blended)

        # matched_colors.append(sampled)

    return torch.stack(matched_colors, dim=0)  # [B, P, 3]

def compute_losses_and_forward_pass(network, data, light_directions, opt, lpips_loss_fn, device, dataset=None, is_training=True):
    """
    Compute forward pass and losses for both training and validation.
    Returns: loss, tree_occ_loss, shadow_occ_loss, bce_occ_loss, color_loss, loss_class, accuracy, visualizations
    """
    orthophoto, gt_mesh, dsm_pc, index_view, shadow_ortho, labels, gt_colors, filenames = data
    
    # # Print the file paths in this batch
    # print(f"📁 Batch file paths:")
    # for i, filename in enumerate(filenames):
    #     print(f"   {i}: {filename}")
    
    labels = labels.cuda()
    orthophoto, gt_mesh, dsm_pc, shadow_ortho = orthophoto.cuda(), gt_mesh.cuda(), dsm_pc.cuda(), shadow_ortho.cuda()
    gt_colors = gt_colors.cuda()
    B, P, _ = gt_mesh.shape

 #   orthophoto = (orthophoto - 0.5) / 0.5

    # gt_colors = histogram_match_gt_colors(gt_mesh, gt_colors, orthophoto)

    index_view = index_view % light_directions.shape[0]
    batch_light_directions = light_directions[index_view].cuda()

    if opt.dsm_convex_hull:
        dsm_pc = sample_points_in_convex_hull(B, dsm_pc, num_query_points=opt.num_points*2) 

    # Query points generation
    if opt.variable == '1':
        full_query_points = create_query_points(B, gt_mesh, num_query_points=opt.num_query).to(device)
    elif opt.variable == '2':
        full_query_points = sample_points_in_convex_hull(B, dsm_pc, num_query_points=opt.num_query)
    elif opt.variable == '3':
        full_query_points = create_normalized_query_points(B, num_query_points=opt.num_query)

    # # Select the first batch for visualization and ensure X is 2D [N, 3]
    # vis.scatter(
    #     X=gt_mesh[0].detach().cpu().numpy(),
    #     win=f"GT-points",
    #     opts=dict(
    #         title=f"GT Occupied Points with Colors-{labels[0].item()}-filename: {filenames[0]})",
    #         markersize=2
    #     )
    # )

    query_points = full_query_points[:B].clone()
    if opt.top_k_gt_occupancy:
        gt_occupancy = compute_occupancy_top_k(B, query_points, gt_mesh, top_k=opt.top_k)
    else: 
        gt_occupancy = compute_occupancy(B, query_points, gt_mesh, threshold=opt.thres) ## TOSEE
    gt_points = extract_threshold_occupied_points(gt_occupancy, query_points, threshold=0.5)

    # Forward pass
    # if opt.model == '10':  # If colors are enabled and not using the new model
    #     occupancy_pred, class_logits = network(dsm_pc, orthophoto, query_points, light_dir=batch_light_directions)        
    #     pred_colors = None
    if opt.colors == True: # and opt.model != '10':
        occupancy_pred, class_logits, pred_colors = network(dsm_pc, orthophoto, query_points)
    else:
        occupancy_pred, class_logits = network(dsm_pc, orthophoto, query_points)
        pred_colors = None
    probs = torch.sigmoid(occupancy_pred).squeeze(-1)

    # Initialize losses
    tree_occ_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    shadow_occ_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    silh_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    color_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    loss_class = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    accuracy = 0.0  # Initialize accuracy

    if opt.colors:
        dists = torch.cdist(query_points, gt_mesh, p=2)  # [B, N, P]
        _, closest_idx = torch.min(dists, dim=2)         # [B, N]
        B, N, _ = query_points.shape
        batch_indices = torch.arange(B, device=gt_mesh.device).unsqueeze(1).expand(-1, N)  # [B, N]
        matched_gt_colors = gt_colors[batch_indices, closest_idx] # [B, N, 3]
        final_colors = matched_gt_colors  # [B, N, 3]     
        weight_color = 100.0  
        if opt.classes_loss: 
            weight_color = 100.0
        color_loss = F.mse_loss(pred_colors, final_colors) * weight_color

        print('Color loss:', color_loss.item())

    # if opt.colors:
    #     dists = torch.cdist(query_points, gt_mesh, p=2)  # [B, N, P]
    #     _, closest_idx = torch.min(dists, dim=2)         # [B, N]
    #     B, N, _ = query_points.shape
    #     batch_indices = torch.arange(B, device=gt_mesh.device).unsqueeze(1).expand(-1, N)  # [B, N]
    #     matched_gt_colors = gt_colors[batch_indices, closest_idx] # [B, N, 3]
    #     final_colors = matched_gt_colors  # already normalized to [0,1]

    #     # Convert to LAB for perceptual loss
    #     pred_lab = rgb_to_lab(pred_colors.permute(0,2,1).contiguous().view(-1,3).view(-1,1,1,3).permute(0,3,1,2))
    #     gt_lab   = rgb_to_lab(final_colors.permute(0,2,1).contiguous().view(-1,3).view(-1,1,1,3).permute(0,3,1,2))

    #     lab_loss = F.mse_loss(pred_lab, gt_lab)
    #     l2_loss = F.mse_loss(pred_colors, final_colors)

    #     weight_color = 0.01 # 100.0 if opt.classes_loss else 50.0
    #     color_loss = (0.7 * lab_loss + 0.3 * l2_loss) * weight_color
    #     print('Color loss:', color_loss.item())


    if opt.classes_loss:
        # # Debug: Check label ranges before computing loss
        # print(f"Class logits shape: {class_logits.shape}, Labels shape: {labels.shape}")
        # print(f"Label values: {labels}")
        # print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")
        # print(f"Number of classes in model: {class_logits.shape[1]}")
        
        # Ensure labels are within valid range [0, num_classes-1]
        num_classes = class_logits.shape[1]
        valid_mask = (labels >= 0) & (labels < num_classes)
        
        if not valid_mask.all():
            print(f"Warning: Found {(~valid_mask).sum().item()} invalid labels")
            print(f"Invalid labels: {labels[~valid_mask]}")
            # Clamp invalid labels to valid range
            labels = torch.clamp(labels, 0, num_classes - 1)
            print(f"Clamped labels: {labels}")
        
        # Compute multi-species classification loss
        # class_logits should have shape [B, num_species] where num_species is the number of unique species
        # labels should have shape [B] with values in range [0, num_species-1]
        class_loss = F.cross_entropy(class_logits, labels)
        loss_class = class_loss * 10.0  # Adjust weight as needed
        
        # # Calculate accuracy for monitoring
        # _, predicted_species = torch.max(class_logits, 1)
        # correct_predictions = (predicted_species == labels).sum().item()
        # accuracy = correct_predictions / labels.size(0)
        
        # if dataset and hasattr(dataset, 'get_label_to_species'):
        #     # Print species information for all samples in the batch
        #     print(f"🌳 Species Information for Batch:")
        #     for i in range(labels.size(0)):
        #         true_species = dataset.get_label_to_species(labels[i].item())
        #         pred_species = dataset.get_label_to_species(predicted_species[i].item())
        #         filename = filenames[i] if i < len(filenames) else f"sample_{i}"
        #         print(f"   {filename}: Ground Truth = {true_species} (label: {labels[i].item()}), Prediction = {pred_species} (label: {predicted_species[i].item()})")
        # print(f'Species Classification - Loss: {class_loss.item():.4f}, Accuracy: {accuracy:.3f}')
    
    # Losses
    if opt.bce:
        tree_occ_loss = occupancy_loss_with_logits(occupancy_pred, gt_occupancy) * 10
    occupied_points, topk_indices = extract_top_k_occupied_points(probs, query_points, top_k=opt.top_k)
    
    # Visualization data
    vis_data = {}
    
    if opt.shadow:
        pred_shadow_img2 = soft_point_projectionShadow(None, query_points, occupancy_pred, 
                                                      image_size=opt.image_size, top_k=opt.top_k, 
                                                      light_directions=batch_light_directions)
        gt_shadow_img2 = soft_point_projectionShadow(None, query_points, gt_occupancy, 
                                                     image_size=opt.image_size, top_k=opt.top_k, 
                                                     light_directions=batch_light_directions, 
                                                     top_k_points=gt_mesh)
        gt_shadow_img2 = gt_shadow_img2.repeat(1, 3, 1, 1)
        pred_shadow_img2 = pred_shadow_img2.repeat(1, 3, 1, 1)
        shadow_occ_loss = lpips_loss_fn(pred_shadow_img2, gt_shadow_img2).mean() * 3 #* 5
        vis_data['shadow'] = (gt_shadow_img2, pred_shadow_img2)

    if opt.silhouettes:
        silhouette_loss = compute_silhouette_losses(query_points, occupancy_pred, gt_occupancy, 
                                                 gt_mesh, lpips_loss_fn, opt)
        silh_loss = silhouette_loss['total_loss'] * 3
        vis_data['silhouettes'] = silhouette_loss['vis_data']

    total_loss = tree_occ_loss + silh_loss + shadow_occ_loss + color_loss + loss_class

    return {
        'loss': total_loss,
        'tree_occ_loss': tree_occ_loss,
        'shadow_occ_loss': shadow_occ_loss,
        'silh_occ_loss': silh_loss,
        'color_loss': color_loss,
        'loss_class': loss_class,
        'accuracy': accuracy,  # Add accuracy to return values
        'gt_points': gt_points,
        'occupied_points': occupied_points,
        'vis_data': vis_data,
        'query_points': query_points,
        'gt_mesh': gt_mesh,
        'gt_occupancy': gt_occupancy,
        'occupancy_pred': occupancy_pred,
        'dsm_pc': dsm_pc,
        'orthophoto': orthophoto,
        'gt_colors': gt_colors,
        'pred_colors': pred_colors,
        'filenames': filenames,
    }

import torchvision.utils as vutils
def save_tensor_as_image(tensor, filepath, normalize=True):
    """
    Save a tensor as an image file.
    Args:
        tensor: Tensor of shape [B, C, H, W] or [C, H, W]
        filepath: Path to save the image
        normalize: Whether to normalize the tensor to [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch
    
    if normalize:
        # Normalize to [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    # Convert to PIL Image and save
    vutils.save_image(tensor, filepath, normalize=False, padding=0)
    print(f"Saved image: {filepath}")
def save_visualization_images(vis_data, epoch, batch_idx, save_dir, mode="train"):
    """
    Save shadow and silhouette visualization images to disk.
    Args:
        vis_data: Dictionary containing visualization data
        epoch: Current epoch number
        batch_idx: Current batch index
        save_dir: Directory to save images
        mode: "train" or "val"
    """
    # Create save directory if it doesn't exist
    vis_save_dir = os.path.join(save_dir, "visualizations", f"epoch_{epoch:03d}")
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # Save shadow images
    if 'shadow' in vis_data:
        gt_shadow, pred_shadow = vis_data['shadow']
        
        gt_path = os.path.join(vis_save_dir, f"{mode}_batch_{batch_idx:03d}_gt_shadow.png")
        pred_path = os.path.join(vis_save_dir, f"{mode}_batch_{batch_idx:03d}_pred_shadow.png")
        
        save_tensor_as_image(gt_shadow, gt_path)
        save_tensor_as_image(pred_shadow, pred_path)
    
    # Save silhouette images
    if 'silhouettes' in vis_data:
        for angle, (gt_silhouette, pred_silhouette) in vis_data['silhouettes'].items():
            gt_path = os.path.join(vis_save_dir, f"{mode}_batch_{batch_idx:03d}_gt_silhouette_{angle}.png")
            pred_path = os.path.join(vis_save_dir, f"{mode}_batch_{batch_idx:03d}_pred_silhouette_{angle}.png")
            
            save_tensor_as_image(gt_silhouette, gt_path)
            save_tensor_as_image(pred_silhouette, pred_path)
    
    print(f"Saved visualization images for {mode} epoch {epoch}, batch {batch_idx}")

def compute_silhouette_losses(query_points, occupancy_pred, gt_occupancy, gt_mesh, lpips_loss_fn, opt):
    """Compute silhouette losses for multiple rotations"""
    rotations = [
        (0, lambda x: x),  # No rotation
        (90, rotate_mesh_y_90),
        (45, rotate_mesh_y_45),
        (135, rotate_mesh_y_135)
    ]
    
    total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    vis_data = {}
    
    for angle, rotate_func in rotations:
        rotated_mesh = rotate_func(gt_mesh)
        rotated_query = rotate_func(query_points)
        
        pred_shadow = soft_point_projection(None, rotated_query, occupancy_pred, 
                                          image_size=int(opt.image_size), top_k=opt.top_k)
        gt_shadow = soft_point_projection(None, rotated_query, gt_occupancy, 
                                        image_size=int(opt.image_size), top_k=opt.top_k, 
                                        top_k_points=rotated_mesh)
        
        pred_shadow = pred_shadow.repeat(1, 3, 1, 1)
        gt_shadow = gt_shadow.repeat(1, 3, 1, 1)
        
        loss = lpips_loss_fn(pred_shadow, gt_shadow).mean()  
        total_loss = total_loss + loss
        
        vis_data[angle] = (gt_shadow, pred_shadow)

    if opt.top_silh:
        pred_shadow = soft_point_projection(None, rotated_query, occupancy_pred, 
                                            image_size=int(opt.image_size), top_k=opt.top_k, axis='z')
        gt_shadow = soft_point_projection(None, rotated_query, gt_occupancy, 
                                        image_size=int(opt.image_size), top_k=opt.top_k, 
                                        top_k_points=rotated_mesh, axis='z')
        pred_shadow = pred_shadow.repeat(1, 3, 1, 1)
        gt_shadow = gt_shadow.repeat(1, 3, 1, 1)
        loss = lpips_loss_fn(pred_shadow, gt_shadow).mean()  
        total_loss = total_loss + loss
        vis_data[360] = (gt_shadow, pred_shadow)

    return {'total_loss': total_loss, 'vis_data': vis_data}

if __name__ == "__main__":
    print(opt)

    # ===========================
    # Setup Logging & Visdom
    # ===========================
    vis = visdom.Visdom(port=opt.port, env=opt.env) # 8090 8091 (my_p2) 8092 

    vis.close(win=None)

    if opt.model == '1': # normal
        from model_normal_categories import TreeReconstructionNet
    elif opt.model == '2': # DSM 
        from model_dsm import TreeReconstructionNet
    elif opt.model == '3': # ortho
        from model_ortho import TreeReconstructionNet
    elif opt.model == '4':
        from model_normal_with_noise import TreeReconstructionNet ## for variety 
    elif opt.model == '10': # dsm with noise
        from model_normal_categories_noRefinement import TreeReconstructionNet
    elif opt.model == '11': 
        from model_colors_old import TreeReconstructionNet
    else: # colors
        from model_colors import TreeReconstructionNet

    save_path = opt.env
    dir_name = os.path.join('./log_p2_final', save_path) # ./log
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # ===========================
    # Load Dataset
    # ===========================
    dataset = TreeDataset(npoints=opt.num_points, num_trees=opt.num_trees, many_trees=opt.many_trees)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, num_workers=int(opt.workers), shuffle=True, collate_fn=collate_fn, drop_last=True)  # drop_last=True to avoid batch size 1

    dataset_test = TreeDataset(npoints=opt.num_points, train=False, num_trees=opt.num_trees, many_trees=opt.many_trees)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, num_workers=int(opt.workers), shuffle=False, collate_fn=collate_fn, drop_last=True)  # drop_last=True to avoid batch size 1

    # Get species information for model initialization
    species_info = dataset.get_species_info()
    num_species = species_info['num_species']
    print(f"Dataset has {num_species} species: {species_info['species_list']}")

    # ===========================
    # Load Model
    # ===========================
    network = TreeReconstructionNet(num_points=opt.num_points, num_species=num_species)
    
    network.cuda()

    network.apply(init_weights_xavier)

    optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Track loss curves
    train_tree_occ_curve, val_tree_occ_curve, train_shadow_occ_curve, val_shadow_occ_curve, train_total_curve, val_total_curve = [], [], [], [], [], []
    train_classes_curve, val_classes_curve = [], []
    train_accuracy_curve, val_accuracy_curve = [], []  # Track classification accuracy
    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()

    # **Initialize Visdom Plots**
    vis.line(
        X=np.array([[0, 0, 0]]),  # Total Loss, Points Loss, Shadow Loss
        Y=np.array([[0, 0, 0]]),
        win='train_losses_plot',
        opts=dict(title="Training Losses", legend=["Total Loss", "Points Loss", "Shadow Loss"], markersize=2)
    )
    vis.line(
        X=np.array([[0, 0, 0]]),
        Y=np.array([[0, 0, 0]]),
        win='val_losses_plot',
        opts=dict(title="Validation Losses", legend=["Total Loss", "Points Loss", "Shadow Loss"], markersize=2)
    )
    vis.line(
        X=np.array([[0, 0]]),  # Train and Validation Total Loss
        Y=np.array([[0, 0]]),
        win='total_loss_plot',
        opts=dict(title="Total Loss", legend=["Train", "Validation"], markersize=2)
    )
    
    # ===========================
    # Load Model & Optimizer
    # ===========================
    start_epoch = 0
    if opt.model_previous_training:
        # network.load_state_dict(torch.load(opt.model_previous_training))
        # ===========================
        # Load Model & Optimizer
        # ===========================
        checkpoint_path = f"{dir_name}/network.pth"
        start_epoch = load_checkpoint(checkpoint_path, network, optimizer)  # Resume training from checkpoint
        plot_save_path = f"{dir_name}/plot_data.json"
        (train_total_curve, val_total_curve,
        train_tree_occ_curve, val_tree_occ_curve,
        train_shadow_occ_curve, val_shadow_occ_curve,
        train_classes_curve, val_classes_curve) = load_curves(plot_save_path)
        print(" Previous weights loaded ")

    # ===========================
    # Training Loop
    # ===========================

    # Load light directions from file
    with open("/home/grammatikakis1/TREES_DATASET/ORTHOPHOTOS/light_directions.txt", "r") as f: # "/home/grammatikakis1/TREES_DATASET/ORTHOPHOTOS/light_directions.txt"
        light_directions_list = [list(map(float, line.strip().split())) for line in f]

    # Convert to a PyTorch tensor
    light_directions = torch.tensor(light_directions_list, dtype=torch.float32).cuda()  # Shape (Total Views, 3)
    
    query_points = None
    
    top_k = opt.top_k # opt.num_points # opt.num_points # 5000 # opt.num_points # 1500 # 5000 # min(opt.num_points, 3500) # opt.num_points # 10000 # opt.num_points  # Number of top occupied points to keep

    lrate = optimizer.param_groups[0]['lr']

    for param in network.parameters():
        param.requires_grad = True

    # perceptual_loss = VGGPerceptualLoss().cuda()
    import lpips
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    shadow_occ_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    tree_occ_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    bce_occ_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    silh_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    color_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    loss_class = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    shadow_gt = None
    shadow_pred = None
    gt_shadow_img, pred_shadow_img = None, None
    from DISTS_pytorch import DISTS
    D = DISTS().to(device)
    for epoch in range(start_epoch, opt.nepoch):
        train_loss.reset()
        network.train()
        last_train_result = None  # Keep track of last result for metrics
        
        for i, data in enumerate(dataloader):
            # Skip batches with size 1 to avoid BatchNorm issues
            if data[0].size(0) == 1:
                print(f"Skipping batch {i} with size 1 to avoid BatchNorm issues")
                continue
                
            result = compute_losses_and_forward_pass(network, data, light_directions, opt, lpips_loss_fn, device, dataset, is_training=True)
            last_train_result = result  # Store for metrics tracking
            
            optimizer.zero_grad()
            result['loss'].backward(retain_graph=True)
            
            if epoch % 10 == 0 and i == 0:
                print('differentiable:', result['shadow_occ_loss'].requires_grad, result['tree_occ_loss'].requires_grad)
                for name, param in network.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name}: grad norm = {grad_norm:.6f}")
            
            optimizer.step()
            train_loss.update(result['loss'].item())

            # print('filenames:', result['filenames'])
            
            # # Print species information for each file in the batch
            # if hasattr(dataset, 'get_species_info') and 'filenames' in result:
            #     print('🌳 Tree species in this batch:')
            #     for idx, filename in enumerate(result['filenames']):
            #         # Try to get the species from the dataset
            #         if hasattr(dataset, 'species_to_label') and dataset.species_to_label:
            #             if filename in dataset.species_to_label:
            #                 species_label = dataset.species_to_label[filename]
            #                 if species_label < len(dataset.species_list):
            #                     species_name = dataset.species_list[species_label]
            #                     print(f'   📁 {filename}: {species_name} (label: {species_label})')
            #                 else:
            #                     print(f'   📁 {filename}: Unknown species (label: {species_label})')
            #             else:
            #                 print(f'   📁 {filename}: Not in species database')
            #         else:
            #             print(f'   📁 {filename}: No species data available')

            print(f"[Epoch {epoch+1}/{opt.nepoch}] Batch {i+1}/{len(dataloader)} Loss: {result['loss'].item():.4f}  -- Tree Loss: {result['tree_occ_loss'].item():.4f} -- Shadow Loss: {result['shadow_occ_loss'].item():.4f} -- Silh Loss: {result['silh_occ_loss'].item():.4f} -- Color Loss: {result['color_loss'].item():.4f} -- Class Loss: {result['loss_class'].item():.4f} -- Accuracy: {result['accuracy']:.3f} -- lr: {optimizer.param_groups[0]['lr']}")

            # Visualization
            if i % 5 == 0: # == len(dataloader) - 1:
                gt_points_vis = [points.cpu().detach().numpy() for points in result['gt_points']]
                occupied_points_vis = [points.cpu().detach().numpy() for points in result['occupied_points']]

                if opt.colors:                    
                    # Get colors for GT occupied points (not all mesh points)
                    gt_points_tensor = result['gt_points'][0]  # GT occupied points [N_gt, 3]
                    gt_mesh_tensor = result['gt_mesh'][0]      # All mesh points [P_mesh, 3]
                    gt_colors_tensor = result['gt_colors'][0]  # Colors for all mesh points [P_mesh, 3]
                    
                    # Find closest mesh points for each GT occupied point to get corresponding colors
                    if len(gt_points_tensor) > 0 and len(gt_mesh_tensor) > 0:
                        dists = torch.cdist(gt_points_tensor.unsqueeze(0), gt_mesh_tensor.unsqueeze(0), p=2)  # [1, N_gt, P_mesh]
                        _, closest_idx = torch.min(dists.squeeze(0), dim=1)  # [N_gt]
                        gt_point_colors = gt_colors_tensor[closest_idx]  # [N_gt, 3]
                        
                        pts = gt_points_vis[0]  # [N_gt, 3]
                        cols = (gt_point_colors.detach().cpu().numpy() * 255).astype(np.uint8)  # [N_gt, 3]
                        
                        vis.scatter(
                            X=pts,
                            win=f"GT-points",
                            opts=dict(
                                title=f"GT Occupied Points with Colors",
                                markercolor=cols,
                                markersize=6
                            )
                        )

                    # Visualize predicted colors for top-k occupied points
                    if 'occupied_points' in result and len(result['occupied_points']) > 0:
                        occupied_points_tensor = result['occupied_points'][0]  # [top_k, 3]
                        
                        # Get predicted colors for the top-k occupied points
                        # We need to find which query points correspond to the occupied points
                        query_points_tensor = result['query_points'][0]  # [N_query, 3]
                        pred_colors_tensor = result['pred_colors'][0]    # [N_query, 3]
                        
                        # Find indices of occupied points in the query points
                        if len(occupied_points_tensor) > 0 and len(query_points_tensor) > 0:
                            # Use the topk_indices from extract_top_k_occupied_points
                            probs = torch.sigmoid(result['occupancy_pred']).squeeze(-1)
                            _, topk_indices = extract_top_k_occupied_points(probs, result['query_points'], top_k=opt.top_k)
                            
                            occupied_pred_colors = pred_colors_tensor[topk_indices[0]]  # [top_k, 3]
                            
                            pts_np = occupied_points_vis[0]  # [top_k, 3]
                            cols_np = (occupied_pred_colors.detach().cpu().numpy() * 255).astype(np.uint8)
                            
                            vis.scatter(
                                X=pts_np,
                                win=f"Occupied-points",
                                opts=dict(
                                    title=f"Occupied Points with Predicted Color",
                                    markercolor=cols_np,
                                    markersize=6
                                )
                            )
                         
                    # print("First 3 GT colors:", gt_colors[0][:3].cpu().numpy())
                    # print("First 3 predicted colors:", pred_colors[0][topk_indices[0][:3]].detach().cpu().numpy())
                
                # Regular visualization
                visualize(network, vis, "Train", result['dsm_pc'], result['query_points'], gt_points_vis, occupied_points_vis, result['orthophoto'], result['gt_mesh'], result['gt_occupancy'], result['occupancy_pred'], result['filenames'])

                # Visualize shadows/silhouettes if enabled
                if 'shadow' in result['vis_data']:
                    gt_shadow, pred_shadow = result['vis_data']['shadow']
                    vis_images(vis, gt_shadow, title='GT-Shadow')
                    vis_images(vis, pred_shadow, title='Pred-Shadow')
                
                if 'silhouettes' in result['vis_data']:
                    for angle, (gt_shadow, pred_shadow) in result['vis_data']['silhouettes'].items():
                        vis_images(vis, gt_shadow, title=f'GT-{angle}')
                        vis_images(vis, pred_shadow, title=f'Pred-{angle}')

                # Save validation visualization images to disk
                if opt.save_images:
                    dir_images_name = os.path.join(dir_name, "visualizations", f"epoch_{epoch:03d}")
                    os.makedirs(dir_images_name, exist_ok=True)
                    save_visualization_images(result['vis_data'], epoch, i, dir_images_name, mode="val")

                
        train_total_curve = np.append(train_total_curve, result['loss'].item())
        train_tree_occ_curve = np.append(train_tree_occ_curve, result['silh_occ_loss'].item()) # Silhouette loss
        train_shadow_occ_curve = np.append(train_shadow_occ_curve, result['shadow_occ_loss'].item()) # Shadow loss
        train_classes_curve = np.append(train_classes_curve, result['loss_class'].item()) # Classification loss
        train_accuracy_curve = np.append(train_accuracy_curve, result['accuracy']) # Classification accuracy
        # train_shadow_occ_curve = np.append(train_shadow_occ_curve, color_loss.item())

        torch.cuda.empty_cache()

        # ===========================
        # Validation
        # ===========================
        val_loss.reset()
        network.eval()
        
        with torch.no_grad():
            last_result = None  # Keep track of last result for loss curves
            for i, data in enumerate(dataloader_test):
                # Skip batches with size 1 to avoid BatchNorm issues
                if data[0].size(0) == 1:
                    print(f"Skipping validation batch {i} with size 1 to avoid BatchNorm issues")
                    continue
                    
                result = compute_losses_and_forward_pass(network, data, light_directions, opt, lpips_loss_fn, device, dataset_test, is_training=False)
                last_result = result  # Store for loss tracking
                
                val_loss.update(result['loss'].item())
                
                print(f"[Validation] Epoch {epoch+1}/{opt.nepoch} Batch {i+1}/{len(dataloader_test)} Loss: {result['loss'].item():.4f} -- Tree Loss: {result['tree_occ_loss'].item():.4f} -- Shadow Loss: {result['shadow_occ_loss'].item():.4f} -- Silh Loss: {result['silh_occ_loss'].item():.4f} -- Color Loss: {result['color_loss'].item():.4f} -- Class Loss: {result['loss_class'].item():.4f} -- Accuracy: {result['accuracy']:.3f}")

                # # Print validation species information
                # if hasattr(dataset_test, 'get_species_info') and 'filenames' in result:
                #     print('🌳 Validation tree species in this batch:')
                #     for idx, filename in enumerate(result['filenames']):
                #         if hasattr(dataset_test, 'species_to_label') and dataset_test.species_to_label:
                #             if filename in dataset_test.species_to_label:
                #                 species_label = dataset_test.species_to_label[filename]
                #                 if species_label < len(dataset_test.species_list):
                #                     species_name = dataset_test.species_list[species_label]
                #                     print(f'   📁 {filename}: {species_name} (label: {species_label})')
                #                 else:
                #                     print(f'   📁 {filename}: Unknown species (label: {species_label})')
                #             else:
                #                 print(f'   📁 {filename}: Not in species database')
                #         else:
                #             print(f'   📁 {filename}: No species data available')

                # Visualization
                if i == len(dataloader_test) - 1:
                    gt_points_vis = [points.cpu().detach().numpy() for points in result['gt_points']]
                    occupied_points_vis = [points.cpu().detach().numpy() for points in result['occupied_points']]
                    
                    if opt.colors:                    
                        # Get colors for GT occupied points (not all mesh points)
                        gt_points_tensor = result['gt_points'][0]  # GT occupied points [N_gt, 3]
                        gt_mesh_tensor = result['gt_mesh'][0]      # All mesh points [P_mesh, 3]
                        gt_colors_tensor = result['gt_colors'][0]  # Colors for all mesh points [P_mesh, 3]
                        
                        # Find closest mesh points for each GT occupied point to get corresponding colors
                        if len(gt_points_tensor) > 0 and len(gt_mesh_tensor) > 0:
                            dists = torch.cdist(gt_points_tensor.unsqueeze(0), gt_mesh_tensor.unsqueeze(0), p=2)  # [1, N_gt, P_mesh]
                            _, closest_idx = torch.min(dists.squeeze(0), dim=1)  # [N_gt]
                            gt_point_colors = gt_colors_tensor[closest_idx]  # [N_gt, 3]
                            
                            pts = gt_points_vis[0]  # [N_gt, 3]
                            cols = (gt_point_colors.detach().cpu().numpy() * 255).astype(np.uint8)  # [N_gt, 3]
                            
                            vis.scatter(
                                X=pts,
                                win=f"GT-points",
                                opts=dict(
                                    title=f"GT Occupied Points with Colors",
                                    markercolor=cols,
                                    markersize=6
                                )
                            )

                        # Visualize predicted colors for top-k occupied points
                        if 'occupied_points' in result and len(result['occupied_points']) > 0:
                            occupied_points_tensor = result['occupied_points'][0]  # [top_k, 3]
                            
                            # Get predicted colors for the top-k occupied points
                            # We need to find which query points correspond to the occupied points
                            query_points_tensor = result['query_points'][0]  # [N_query, 3]
                            pred_colors_tensor = result['pred_colors'][0]    # [N_query, 3]
                            
                            # Find indices of occupied points in the query points
                            if len(occupied_points_tensor) > 0 and len(query_points_tensor) > 0:
                                # Use the topk_indices from extract_top_k_occupied_points
                                probs = torch.sigmoid(result['occupancy_pred']).squeeze(-1)
                                _, topk_indices = extract_top_k_occupied_points(probs, result['query_points'], top_k=opt.top_k)
                                
                                occupied_pred_colors = pred_colors_tensor[topk_indices[0]]  # [top_k, 3]
                                
                                pts_np = occupied_points_vis[0]  # [top_k, 3]
                                cols_np = (occupied_pred_colors.detach().cpu().numpy() * 255).astype(np.uint8)
                                
                                vis.scatter(
                                    X=pts_np,
                                    win=f"Occupied-points",
                                    opts=dict(
                                        title=f"Occupied Points with Predicted Color",
                                        markercolor=cols_np,
                                        markersize=6
                                    )
                                )
                    
                    # Regular visualization
                    visualize(network, vis, "Val", result['dsm_pc'], result['query_points'], gt_points_vis, occupied_points_vis, result['orthophoto'], result['gt_mesh'], result['gt_occupancy'], result['occupancy_pred'], result['filenames'])
                    
    
                    # Visualize shadows/silhouettes if enabled
                    if 'shadow' in result['vis_data']:
                        gt_shadow, pred_shadow = result['vis_data']['shadow']
                        vis_images(vis, gt_shadow, title='GT-Shadow')
                        vis_images(vis, pred_shadow, title='Pred-Shadow')
                    
                    if 'silhouettes' in result['vis_data']:
                        for angle, (gt_shadow, pred_shadow) in result['vis_data']['silhouettes'].items():
                            vis_images(vis, gt_shadow, title=f'GT-{angle}')
                            vis_images(vis, pred_shadow, title=f'Pred-{angle}')

        scheduler.step(val_loss.avg)
        # Use individual loss components from last batch for curves
        val_total_curve = np.append(val_total_curve, last_result['loss'].item() if last_result else val_loss.avg)
        val_tree_occ_curve = np.append(val_tree_occ_curve, last_result['silh_occ_loss'].item() if last_result else val_loss.avg)
        val_shadow_occ_curve = np.append(val_shadow_occ_curve, last_result['shadow_occ_loss'].item() if last_result else val_loss.avg)
        val_classes_curve = np.append(val_classes_curve, last_result['loss_class'].item() if last_result else val_loss.avg)
        val_accuracy_curve = np.append(val_accuracy_curve, last_result['accuracy'] if last_result else 0.0)
        
        # ===========================
        # Log & Save Model
        # ===========================
        log_table = {
            "train_loss": train_loss.avg,
            "val_loss": val_loss.avg,
            "train_class_loss": last_train_result['loss_class'].item() if last_train_result else 0.0,
            "val_class_loss": last_result['loss_class'].item() if last_result else 0.0,
            "train_accuracy": last_train_result['accuracy'] if last_train_result else 0.0,
            "val_accuracy": last_result['accuracy'] if last_result else 0.0,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr']
        }


        print(log_table)
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(log_table) + '\n')

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,  # Save next epoch number
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        torch.save(checkpoint, f'{dir_name}/network.pth') # _epoch_{epoch+1}.pth') network.state_dict()

        # Save loss curves for later reloading
        plot_data = {
            "train_total_curve": train_total_curve.tolist(),
            "val_total_curve": val_total_curve.tolist(),
            "train_tree_occ_curve": train_tree_occ_curve.tolist(),
            "val_tree_occ_curve": val_tree_occ_curve.tolist(),
            "train_shadow_occ_curve": train_shadow_occ_curve.tolist(),
            "val_shadow_occ_curve": val_shadow_occ_curve.tolist()
        }

        plot_save_path = f"{dir_name}/plot_data.json"
        with open(plot_save_path, 'w') as f:
            json.dump(plot_data, f)

        # ===========================
        # Plot Losses in Visdom
        # ===========================
        vis.line(
                X=np.column_stack((np.arange(len(train_total_curve)), np.arange(len(val_total_curve)))),
                Y=np.log(np.column_stack((np.array(train_total_curve) + 1e-8, np.array(val_total_curve) + 1e-8))),
                win='total_loss_plot',
                opts=dict(title="Total Loss", legend=["Train", "Validation"], markersize=2),
                update='replace'  # Ensure it updates smoothly
            )

        # ===========================
        # Plot Tree Occupancy Loss in Visdom
        # ===========================
        vis.line(
            X=np.column_stack((np.arange(len(train_total_curve)), np.arange(len(train_tree_occ_curve)), np.arange(len(train_shadow_occ_curve)))),
            Y=np.log(np.column_stack((np.array(train_total_curve) + 1e-8, np.array(train_tree_occ_curve) + 1e-8, np.array(train_shadow_occ_curve) + 1e-8))),
            win='train_losses_plot',
            opts=dict(title="Training Losses", legend=["Total Loss", "Points Loss", "Shadow Loss"], markersize=2),
            update='replace'
        )

        # ===========================
        # Plot Shadow Occupancy Loss in Visdom
        # ===========================
        vis.line(
            X=np.column_stack((np.arange(len(val_total_curve)), np.arange(len(val_tree_occ_curve)), np.arange(len(val_shadow_occ_curve)))),
            Y=np.log(np.column_stack((np.array(val_total_curve) + 1e-8, np.array(val_tree_occ_curve) + 1e-8, np.array(val_shadow_occ_curve) + 1e-8))),
            win='val_losses_plot',
            opts=dict(title="Validation Losses", legend=["Total Loss", "Points Loss", "Shadow Loss"], markersize=2),
            update='replace'
        )


    print("Training Complete!")

# CUDA_VISIBLE_DEVICES=5 python B5_colors.py --port 8099 --image_size 96 --batchSize 16 --num_points 2500 --num_query 15000 --num_trees 700 --top_k 2500 --deciduous true --thres 25 --env mixed_colors --model 3 --top_k_shadows 2500 --shadow true --silhouettes true --model_previous_training true