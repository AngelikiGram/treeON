import os
import struct
import sys
sys.path.append('./auxiliary/')
sys.path.append('./auxiliary/dataset/')
sys.path.append('./auxiliary/models/')
from query_points_gen import *
from dataset_test_one_only import TreeDatasetOneOnly
import torch
import numpy as np

def save_points_as_ply(points, filename, colors=None):
    """
    Saves the given points (tensor) as a PLY file.
    points: (N, 3) torch.Tensor or ndarray.
    colors: (N, 3) torch.Tensor or ndarray, values in [0, 255] or [0, 1].
    filename: output path (.ply).
    """
    # Normalize input
    if isinstance(points, list):
        if len(points) == 1 and isinstance(points[0], torch.Tensor):
            points = points[0]  # unwrap
        else:
            points = np.array([
                p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                for p in points
            ])
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # Ensure points are (N,3)
    points = np.asarray(points).reshape(-1, 3)

    print(f"Saving {points.shape[0]} points to {filename}")
    N = points.shape[0]

    # Prepare colors
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        colors = np.asarray(colors).reshape(-1, 3)

        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Write PLY
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {N}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

        for i in range(N):
            x, y, z = points[i].tolist()
            if colors is not None:
                r, g, b = colors[i].tolist()
                f.write(f'{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n')
            else:
                f.write(f'{x:.8f} {y:.8f} {z:.8f}\n')

from occupancy_compute import extract_top_k_occupied_points, compute_occupancy, extract_threshold_occupied_points, compute_occupancy_top_k

def generate_output_pointcloud(tree_id, dataset_root, output_root, model_path=None, model='5', num_points=2500, env='p2_val', variable='3', num_query_points=25000):
    """
    For a given tree_id, load DSM, ortho, tree using your dataset logic,
    run your model to generate output point cloud, and save as output_{tree_id}.ply in output_root.
    """
    import torch

    import visdom
    viz = visdom.Visdom(env=env, port=8099)

    # Import model based on model type
    if args.model == '1': # normal
        from model_normal_categories import TreeReconstructionNet
    elif args.model == '2': # DSM 
        from model_dsm import TreeReconstructionNet
    elif args.model == '3': # ortho
        from model_ortho import TreeReconstructionNet
    elif args.model == '4':
        from model_normal_with_noise import TreeReconstructionNet ## for variety 
    elif args.model == '10': # dsm with noise
        from model_normal_categories_noRefinement import TreeReconstructionNet
    else: # colors
        from model_colors import TreeReconstructionNet

    dataset = TreeDatasetOneOnly(dataset_root, tree_id)
    # Find sample with matching tree_id
    sample = None
    for s in dataset:
        print('Checking sample with tree_id:', s.get('tree_id', None), 'against:', tree_id)
        if s.get('tree_id', None) == "tree_" + tree_id or s.get('tree_id', None) == tree_id:
            sample = s
            break
    if sample is None:
        raise ValueError(f"Tree ID {tree_id} not found in dataset.")
    dsm = sample['dsm']
    ortho = sample['ortho']
    tree = sample['tree']
    # query_points = sample['query_points']

    # Query points generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 1
    if args.variable == '1':
        query_points = create_query_points(B, dsm, num_query_points=args.num_query_points).to(device)
    elif args.variable == '2':
        query_points = sample_points_in_convex_hull(B, dsm, num_query_points=args.num_query_points)
    elif args.variable == '3':
        query_points = create_normalized_query_points(B, num_query_points=args.num_query_points)

    # 2. Load model
    model = TreeReconstructionNet(num_points=args.num_points, num_species=14).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    dsm = dsm.to(device)
    ortho = ortho.to(device)
    query_points = query_points.to(device)
    checkpoint = torch.load(f"./log_p2_final/{args.env}/network.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        # 3. Run inference
        print('dsm shape:', dsm.shape, 'ortho shape:', ortho.shape, 'query points shape:', query_points.shape)
        output_occ, _, pred_colors = model(dsm, ortho, query_points)
        # # 4. Select points with occupancy > 0.5
        # mask = (output_occ.squeeze(-1) > 0.5)
        # output_points = query_points[mask].cpu().numpy()

        probs = torch.sigmoid(output_occ)
        pred_points, topk_indices = extract_top_k_occupied_points(probs, query_points, args.num_points)
        
    # 5. Save output points
    output_path = os.path.join(output_root, f"output_{tree_id}.ply")
    # If you have colors, pass them here as 'colors=...'
    save_points_as_ply(pred_points, output_path, colors=pred_colors)
    print(f"Saved: {output_path}")

    # # Prepare data
    # query_np = query_points.detach().cpu().numpy().reshape(-1, 3)
    # dsm_np = dsm.detach().cpu().numpy().reshape(-1, 3)
    # if isinstance(pred_points, torch.Tensor):
    #     gen_np = pred_points.detach().cpu().numpy().reshape(-1, 3)
    # elif isinstance(pred_points, list) and len(pred_points) > 0 and isinstance(pred_points[0], torch.Tensor):
    #     gen_np = np.stack([p.detach().cpu().numpy() for p in pred_points]).reshape(-1, 3)
    # else:
    #     gen_np = np.asarray(pred_points).reshape(-1, 3)
    # # Stack all points and create labels
    # all_points = np.concatenate([query_np, dsm_np, gen_np], axis=0)
    # labels = np.array([0]*len(query_np) + [1]*len(dsm_np) + [2]*len(gen_np))
    # # Assign a unique color to each point
    # query_color = np.tile([0, 0, 1], (len(query_np), 1))   # Blue
    # dsm_color = np.tile([0, 1, 0], (len(dsm_np), 1))       # Green
    # gen_color = np.tile([1, 0, 0], (len(gen_np), 1))       # Red
    # colors = np.concatenate([query_color, dsm_color, gen_color], axis=0)
    # # Plot with legend toggling
    # viz.scatter(X=all_points, Y=labels+1, opts=dict(
    #     markersize=3,
    #     legend=['Query Points', 'DSM', 'Generated'],
    #     markercolor=colors,
    #     title='Query Points, DSM, Generated',
    #     xlabel='X', ylabel='Y', zlabel='Z',
    #     dim=3
    # ))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate output point cloud for a single tree ID.")
    parser.add_argument("tree_id", type=str, help="Tree ID (e.g. tree_0005)")
    parser.add_argument("--dataset_root", type=str, default="/home/grammatikakis1/TREES_DATASET", help="Path to dataset root")
    parser.add_argument("--output_root", type=str, default="/home/grammatikakis1/COMPARISONS_MODELS/MINE/", help="Path to output folder")
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument("--model", type=str, default="5", help="Model type: 1-normal, 2-dsm, 3-ortho, 4-variety, 10-dsm with noise, else-colors")
    parser.add_argument("--num_points", type=int, default=4000, help="Number of input points")
    parser.add_argument("--env", type=str, default="p2_val", help="Environment name for loading model")
    parser.add_argument("--variable", type=str, default="3", help="Query points generation method: 1-dsm, 2-convex hull, 3-uniform in unit cube")
    parser.add_argument("--num_query_points", type=int, default=85000, help="Number of query points")
    args = parser.parse_args()
    generate_output_pointcloud(args.tree_id, args.dataset_root, args.output_root, args.model_path, args.model, args.num_points, args.env, args.variable, args.num_query_points)



# python validation/generate_output_pointcloud.py tree_0005 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all