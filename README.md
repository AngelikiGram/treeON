# Network Tree Generation (P2)

nohup `CUDA_VISIBLE_DEVICES=0 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 1200 --top_k 2500 --thres 25 --env mixed_noClasses --bce true --shadow true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --top_k_gt_occupancy true --dsm_convex_hull true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=1 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 1200 --top_k 2500 --thres 25 --env mixed_noClasses_thres --bce true --shadow true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --dsm_convex_hull true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=2 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 1200 --top_k 2500 --thres 25 --env mixed_classes --bce true --classes_loss true --shadow true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --dsm_convex_hull true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=3 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 1200 --top_k 2500 --thres 25 --env mixed_classes_thres --bce true --shadow true --classes_loss true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --dsm_convex_hull true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 700 --top_k 2500 --thres 25 --env test --bce true --shadow true --classes_loss true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --dsm_convex_hull true > out.log 2>&1` &

# diff class losses 
nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 700 --top_k 2500 --thres 25 --env test1 --bce true --shadow true --classes_loss true --silhouettes true --model 1 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --dsm_convex_hull true > out.log 2>&1` &

A neural network-based system for generating 3D tree models with realistic rendering capabilities, including shadows, silhouettes, and color information.

## Current Experiments

**Variable 3**: Normalized query points for improved model performance

### Active Training Configurations
1. **mixed_classes**: top_k_gt_occupancy (no threshold, uses top_k points for more coverage) + classes + no noise in model
2. **mixed_noClasses**: Standard configuration without class loss
3. **mixed_classes_thres**: Threshold-based + classes for controlled point selection

## Research Links
- [Tree Reconstruction Study](https://studies.cg.tuwien.ac.at/crowdsource?study=treesReconstruction) 


## Features

- **3D Tree Generation**: Generate realistic tree models using neural networks
- **Multi-modal Training**: Support for colors, shadows, silhouettes, and orthographic effects
- **Point Cloud Processing**: Advanced point cloud manipulation and query point generation
- **Blender Integration**: Automated rendering pipeline using Blender
- **Real-time Visualization**: Visdom integration for training monitoring
- **CUDA Acceleration**: GPU-accelerated training and inference

## Project Structure

```
network-tree-gen/
├── train.py                   # Main training script
├── convert_mat_to_obj.py      # Convert MAT files to OBJ format
├── auxiliary/                 # Core utilities and models
│   ├── models/               # Neural network architectures
│   ├── dataset/              # Dataset handling
│   ├── loss_functions.py     # Custom loss functions
│   ├── visualize.py          # Visualization utilities
│   └── utils.py              # General utilities
├── extension/                # CUDA extensions for Chamfer distance
├── landmarks_austria/        # Austrian landmarks dataset
├── validation/               # Validation and rendering scripts
└── log/                      # Training logs
```

## Requirements

- Python 3.11+
- PyTorch with CUDA support
- Blender 4.3
- Visdom
- NumPy, scikit-image
- CUDA-capable GPU



## Installation

### 1. Install Python Dependencies

```bash
pip install torch torchvision numpy scikit-image visdom matplotlib lpips pandas trimesh
```

### 2. Install Blender Dependencies

```bash
"C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin\python.exe" -m pip install scikit-image
```

#### Blender Setup for Dependencies

When working with Blender scripts, ensure proper module imports:

```python
import sys
import site
site_path = site.getusersitepackages()
if site_path not in sys.path:
    sys.path.append(site_path)
import os
import numpy as np
from skimage import io, exposure
```

### 3. Compile CUDA Extensions

Navigate to the `extension/` directory and compile the Chamfer distance CUDA extension:

#### Step 1: Set CUDA_HOME Environment Variable

First, find your CUDA installation:
```bash
# Find CUDA installation directory
find /usr -name "cuda*" -type d 2>/dev/null | head -10
```

Then set the CUDA_HOME environment variable:
```bash
# For CUDA 10.0 (adjust version as needed)
export CUDA_HOME=/usr/local/cuda-10.0
echo 'export CUDA_HOME=/usr/local/cuda-10.0' >> ~/.bashrc
source ~/.bashrc

pip install --no-build-isolation -e .
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'

# Verify CUDA installation
nvcc --version
```

#### Step 2: Compile Extensions
```bash
cd extension
python setup.py build_ext --inplace
```

If you encounter issues, try:
```bash
# Alternative compilation method
CUDA_HOME=/usr/local/cuda-10.0 python setup.py build_ext --inplace
```

## Usage

### Setting up Remote Connection (Optional)

For remote training on GPU servers:

```bash
ssh -L 8097:localhost:8097 -L 8098:localhost:8098 -L 8099:localhost:8099 -L 8090:localhost:8090 -L 8091:localhost:8091 -L 8092:localhost:8092 -p 31415 grammatikakis1@dgxa100.icsd.hmu.gr
source ~/.bashrc
```

### Starting Visdom Server

```bash
python -m visdom.server -port 8099
```

### Training

#### Basic Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --port 8099 \
    --image_size 90 \
    --batchSize 16 \
    --num_points 2500 \
    --num_query 13000 \
    --num_trees 600 \
    --top_k 2500 \
    --deciduous true \
    --thres 25 \
    --env mixed_colors_orthoEffect \
    --bce true \
    --shadow true \
    --silhouettes true \
    --colors true \
    --classes_loss true \
    --model 6 \
    --top_k_shadows 2500 \
    --model_previous_training true \
    --nepoch 700 \
    --variable 3

nohup `CUDA_VISIBLE_DEVICES=0 python train.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 17000 --num_trees 1200 --top_k 2500 --thres 25 --env mixed_classes --bce true --shadow true --silhouettes true --classes_loss true --model 3 --top_k_shadows 2500 --model_previous_training true --nepoch 700 --variable 3 --top_k_gt_occupancy true > out.log 2>&1` &
```

**Stop Training:**
```bash
pkill -f "python train.py"
```

### Rendering with Blender

Generate renderings using Blender:

```bash
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings.py
```

### Key Parameters

- `--variable 3`: Use normalized query points
- `--num_trees`: Total number of trees in the dataset
- `--deciduous`: Include deciduous trees (true/false)
- `--shadow`: Enable shadow generation
- `--silhouettes`: Enable silhouette processing
- `--colors`: Enable color information
- `--model`: Model architecture variant (1-6)

## Validation

Run validation pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py \
    --env test3_norm01_colorsrgb \
    --num_query_points 85000 \
    --top_k 4000 \
    --num_points 4000 \
    --model 4 \
    --deciduous true \
    --variable 2 \
    --top_k_max 1200

CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py --env mixed_noClasses_thres --num_query_points 85000 --top_k 4000 --num_points 4000 --model 1 --deciduous true --variable 3 --top_k_max 12000
```

## Dataset

The project uses Austrian landmarks dataset located in `landmarks_austria/` directory:

- `trees-data.csv`: Tree metadata
- `DATA_LANDMARKS/`: Main dataset directory
- `DSM/`: Digital Surface Models
- `ORTHOPHOTOS/`: Orthographic images
- `TREE_MODELS/`: 3D tree model files

## Models

The system supports multiple model architectures (models 1-6) with different capabilities:

- **Model 6**: Advanced model with color, shadow, and silhouette support
- **Model 4**: Validation-optimized model
- Support for both deciduous and coniferous trees

## Output

Training outputs are saved in:
- `log/`: Training logs and metrics
- `val_npy/`: Validation results in NumPy format
- Generated point clouds and renderings

## Development Workflow

### Syncing Files to Remote Servers

#### To HMU Server
```bash
# Sync project files (excluding .git)
rsync -avz -e "ssh -p 31415" --exclude='.git' /mnt/c/Users/mmddd/Documents/network-tree-gen/ grammatikakis1@dgxa100.icsd.hmu.gr:~/network-tree-gen/
```

#### To TU Wien Atlas Server
```bash
# SSH connection with port forwarding
ssh -L 8097:localhost:8097 -L 8098:localhost:8098 -L 8099:localhost:8099 -L 8090:localhost:8090 -L 8091:localhost:8091 -L 8092:localhost:8092 agrammat@atlas.cg.tuwien.ac.at


ssh -L 8097:localhost:8097 -L 8098:localhost:8098 -L 8099:localhost:8099 -L 8090:localhost:8090 -L 8091:localhost:8091 -L 8092:localhost:8092 -p 31415 grammatikakis1@dgxa100.icsd.hmu.gr

# Sync project files
rsync -avz -e "ssh -p 22" --exclude='.git' /mnt/c/Users/mmddd/Documents/network-tree-gen/ agrammat@atlas.cg.tuwien.ac.at:~/Desktop/network-tree-gen/

# Sync dataset
rsync -avz -e "ssh -p 22" /mnt/d/TREES_DATASET/TREES_DATASET.tar agrammat@atlas.cg.tuwien.ac.at:~/Desktop/

rsync -avz -e "ssh -p 31415" /mnt/c/Users/mmddd/Documents/PUNet_DATASET.tar grammatikakis1@dgxa100.icsd.hmu.gr:~/
```

### Process Management

```bash
# Check running processes
ps aux | grep grammat

# Kill training process
pkill -f "python train.py"

# Source environment
source ~/.bashrc
```

## Troubleshooting

### CUDA Setup Issues

**Problem**: `OSError: CUDA_HOME environment variable is not set`

**Solution**:
```bash

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c conda-forge cudatoolkit-dev

pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Compile with explicit CUDA_HOME
cd extension
CUDA_HOME=/usr/local/cuda-10.0 python setup.py build_ext --inplace
```
