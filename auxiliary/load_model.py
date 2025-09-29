import os
import torch
import json
import numpy as np

def load_checkpoint(checkpoint_path, network, optimizer):
    if os.path.exists(checkpoint_path):
        print('Loading checkpoint from', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # Resume from saved epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0  # Start from scratch if no checkpoint found
        print("Starting training from scratch")

    return start_epoch

def load_curves(plot_save_path):
    if os.path.exists(plot_save_path):
        with open(plot_save_path, 'r') as f:
            plot_data = json.load(f)

        # Convert lists back to NumPy arrays for plotting
        train_total_curve = np.array(plot_data.get("train_total_curve", []))
        val_total_curve = np.array(plot_data.get("val_total_curve", []))
        train_tree_occ_curve = np.array(plot_data.get("train_tree_occ_curve", []))
        val_tree_occ_curve = np.array(plot_data.get("val_tree_occ_curve", []))
        train_shadow_occ_curve = np.array(plot_data.get("train_shadow_occ_curve", []))
        val_shadow_occ_curve = np.array(plot_data.get("val_shadow_occ_curve", []))
        train_classes_curve = np.array(plot_data.get("train_classes_curve", []))
        val_classes_curve = np.array(plot_data.get("val_classes_curve", []))

        print(f"Previous loss curves loaded from {plot_save_path}")

    else:
        print("🆕 No previous loss plots found, starting fresh.")
        train_total_curve, val_total_curve = [], []
        train_tree_occ_curve, val_tree_occ_curve = [], []
        train_shadow_occ_curve, val_shadow_occ_curve = [], []
        train_classes_curve, val_classes_curve = [], []

    return (train_total_curve, val_total_curve,
            train_tree_occ_curve, val_tree_occ_curve,
            train_shadow_occ_curve, val_shadow_occ_curve,
            train_classes_curve, val_classes_curve)