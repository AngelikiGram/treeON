import torch 
import torch.nn.functional as F

def visualize(network, vis, train_val, dsm_pc, query_points, gt_mesh, occupied_points, orthophoto, ground_truth, gt_occupancy, occupancy_pred, filenames, shadow_gt = None, shadow_pred = None):
    """
    Visualize the input data and the output predictions using Visdom."
    """
    train_val = "Train" if network.training else "Val"
#  gt_mesh_np = gt_mesh.transpose(1, 2)[0].detach().cpu().numpy().T  # Shape (3, num_points)
# vis.scatter(
#     X=gt_points[0],
#     win='occupied_points_vis',
#     opts=dict(
#         title="GT Occupied Query Points",
#         markersize=2,
#         markercolor=np.array([[255, 0, 0]] * len(gt_points[0]))  # Red color
#     )
# )

    # Ground Truth Mesh
    ground_truth_mesh_np = ground_truth[0] # .detach().cpu().numpy()
    ground_truth_mesh_np = ground_truth_mesh_np[:, [0, 2, 1]]
    vis.scatter(
        X=ground_truth_mesh_np,
        win=f'{train_val}-GROUND_Mesh', # -{filenames[0]}',
        opts=dict(title=f'{train_val}-GROUND Mesh', markersize=2) # -{filenames[0]}
    )

    # Visdom scatter plot for DSM point cloud
    dsm_pc_np = dsm_pc[0]
    dsm_pc_np = dsm_pc_np[:, [0, 2, 1]]
    vis.scatter(
        X=dsm_pc_np,
        win=f'{train_val}-DSM_PC',
        opts=dict(title=f'{train_val}-DSM Point Cloud', markersize=2)
    )

    # # Visdom scatter plot for Query Points
    # query_points_np = query_points[0].detach().cpu().numpy()  # Shape: (Q, 3)
    # vis.scatter(
    #     X=query_points_np,
    #     win=f'{train_val}-Query_Points',
    #     opts=dict(title=f'{train_val}-Query Points', markersize=2)
    # )

    # Ground Truth Mesh
    gt_mesh_np = gt_mesh[0] # .detach().cpu().numpy()
    gt_mesh_np = gt_mesh_np[:, [0, 2, 1]]
    vis.scatter(
        X=gt_mesh_np,
        win=f'{train_val}-GT_Mesh',
        opts=dict(title=f'{train_val}-Ground Truth Mesh', markersize=2)
    )

    # Predicted Occupied Points
    # print('occupied_points:', occupied_points)
    pred_mesh_np = occupied_points[0]  # Extracted from occupancy prediction
    pred_mesh_np = pred_mesh_np[:, [0, 2, 1]]
    vis.scatter(
        X=pred_mesh_np,
        win=f'{train_val}-Pred_Mesh',
        opts=dict(title=f'{train_val}-Predicted Mesh', markersize=2)
    )

    # Visualize Orthophoto
    orthophoto_np = orthophoto[0].detach().cpu().numpy()
    orthophoto_np = (orthophoto_np - orthophoto_np.min()) / (orthophoto_np.max() - orthophoto_np.min() + 1e-8)
    
    # Include filename in title if available
    orthophoto_title = "Orthophoto"
    if filenames is not None and len(filenames) > 0:
        orthophoto_title = f"Orthophoto - {filenames[0]}"
    
    vis.image(orthophoto_np,
            win='Orthophoto',
            opts=dict(title=orthophoto_title, caption="Input Image"))
       
    if shadow_gt is not None and shadow_pred is not None: 
        # # if shadow_gt.shape[1] == 2 or shadow_gt.shape[1] == 3:            
        # # Ground Truth Shadow
        # vis.scatter(
        #     X=shadow_gt[0],
        #     win=f'{train_val}-GT_Shadow',
        #     opts=dict(title=f'{train_val}-Ground Truth Shadow', markersize=2)
        # )

        # # Predicted Shadow
        # vis.scatter(
        #     X=shadow_pred[0],
        #     win=f'{train_val}-Pred_Shadow',
        #     opts=dict(title=f'{train_val}-Predicted Shadow', markersize=2)
        # )
        # # else:
        # #     # Select first sample
        # #     shadow_pred_vis = shadow_pred[0].unsqueeze(0)  # (1, H, W)
        # #     shadow_gt_vis   = shadow_gt[0].unsqueeze(0)    # (1, H, W)

        # #     # Normalize for display (optional)
        # #     shadow_pred_vis = shadow_pred_vis.float() / shadow_pred_vis.max()
        # #     shadow_gt_vis   = shadow_gt_vis.float() / shadow_gt_vis.max()

        # #     # Show in Visdom
        # #     vis.image(shadow_pred_vis, win='shadow_pred', opts=dict(title='Shadow Prediction'))
        # #     vis.image(shadow_gt_vis, win='shadow_gt', opts=dict(title='Shadow Ground Truth'))
        # #     # shadow = shadow_gt[0, 0].detach().cpu().numpy()
        # #     # vis.image(shadow, win="gt_shadow", opts=dict(title='Ground Truth Shadow', caption='shadow'))
        # #     # shadow = shadow_pred[0, 0].detach().cpu().numpy()  # (H, W)
        # #     # vis.image(shadow, win="pred_shadow", opts=dict(title='Predicted Shadow', caption='shadow'))

        # Visualize Orthophoto
        shadow_gt = F.interpolate(shadow_gt, size=(256, 256), mode='bilinear', align_corners=False)
        shadow_gt_np = shadow_gt[0].detach().cpu().numpy()
        shadow_gt_np = (shadow_gt_np - shadow_gt_np.min()) / (shadow_gt_np.max() - shadow_gt_np.min() + 1e-8)
        vis.image(shadow_gt_np,
                win='GT-Side-2',
                opts=dict(title="GT-Side-2", caption="GT-Side-2"))
    
        # Visualize Orthophoto
        shadow_pred = F.interpolate(shadow_pred, size=(256, 256), mode='bilinear', align_corners=False)
        shadow_pred_np = shadow_pred[0].detach().cpu().numpy()
        shadow_pred_np = (shadow_pred_np - shadow_pred_np.min()) / (shadow_pred_np.max() - shadow_pred_np.min() + 1e-8)
        vis.image(shadow_pred_np,
                win='Pred-Side-2',
                opts=dict(title="Pred-Side-2", caption="Pred-Side-2"))

    # visualize_occupied_predictions_vs_gt(
    #     vis,
    #     predicted_points=occupied_points,
    #     query_points=query_points,
    #     gt_occupancy=gt_occupancy,
    #     title=f'{train_val}-Predicted vs GT Occupancy',
    #     pred_logits=occupancy_pred,
    #     threshold=best_threshold_values
    # )


import torch
import numpy as np

import torch
import numpy as np

def visualize_occupied_predictions_vs_gt(
    vis,
    predicted_points,
    query_points,
    gt_occupancy,
    threshold=0.5,
    title="Occupancy Prediction Accuracy",
    pred_logits=None
):
    """
    Visualize predicted occupied points:
    - 🔴 Red = True Positives (TP)     → predicted = 1 and GT = 1
    - 🔵 Blue = False Positives (FP)   → predicted = 1 but GT = 0
    - 🟢 Green = False Negatives (FN)  → predicted = 0 but GT = 1
    """
    threshold = threshold[0]

    # Get query points and ground truth
    all_query = query_points[0].detach().cpu()        # (N, 3)
    gt_occ = gt_occupancy[0].detach().cpu().squeeze() # (N,)
    
    # Make sure logits are provided
    if pred_logits is None:
        raise ValueError("Need `pred_logits` to visualize prediction accuracy.")

    pred_logits = pred_logits[0].detach().cpu().squeeze()
    pred_probs = torch.sigmoid(pred_logits)

    # Apply threshold
    pred_binary = (pred_probs > threshold).int()
    gt_binary = (gt_occ > threshold).int()

    # Create masks
    TP_mask = (pred_binary == 1) & (gt_binary == 1)
    FP_mask = (pred_binary == 1) & (gt_binary == 0)
    FN_mask = (pred_binary == 0) & (gt_binary == 1)

    # Extract points per category
    TP_points = all_query[TP_mask]
    FP_points = all_query[FP_mask]
    FN_points = all_query[FN_mask]

    # Confidence printing
    print(f"\n🔴 RED (TP): {TP_points.shape[0]} points")
    print(pred_probs[TP_mask])

    print(f"\n🔵 BLUE (FP): {FP_points.shape[0]} points")
    print(pred_probs[FP_mask])

    print(f"\n🟢 GREEN (FN): {FN_points.shape[0]} points (should be low confidence)")
    print(pred_probs[FN_mask])

    print('Sum1:', TP_points.shape[0] + FP_points.shape[0])
    print('Sum2:', TP_points.shape[0] + FP_points.shape[0] + FN_points.shape[0])

    print_average_fp_value(pred_logits, FP_mask)
    print_average_fp_value(pred_logits, FN_mask)
    print_average_fp_value(pred_logits, TP_mask)

    # Stack all points
    all_vis_pts = torch.cat([TP_points, FP_points, FN_points], dim=0)
    labels = torch.tensor(
        [0] * TP_points.shape[0] +
        [1] * FP_points.shape[0] +
        [2] * FN_points.shape[0]
    )

    # Colors (BGR format)
    colors = torch.tensor(
        [[255, 0, 0]] * TP_points.shape[0] +   # 🔴 TP - Red
        [[0, 0, 255]] * FP_points.shape[0] +   # 🔵 FP - Blue
        [[0, 255, 0]] * FN_points.shape[0]     # 🟢 FN - Green
    ).numpy()

    vis.scatter(
        X=all_vis_pts,
        Y=labels + 1,  # Visdom labels must be >= 1
        win=title,
        opts=dict(
            title=title,
            markersize=3,
            markercolor=colors,
            legend=["TP", "FP", "FN"]
        )
    )

def print_average_fp_value(logits, false_positive_mask):
    probs = torch.sigmoid(logits)

    fp_values = probs[false_positive_mask]

    if fp_values.numel() > 0:
        avg_fp = fp_values.mean().item()
        print(f"🟦 Avg FP value (pred > 0, GT = 0): {avg_fp:.4f}")
    else:
        print("🟦 No false positives in this batch.")