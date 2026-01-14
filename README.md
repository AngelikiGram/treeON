# TreeON: Reconstructing 3D Tree Point Clouds from Orthophotos and Heightmaps

This repository contains the complete implementation of **TreeON**, a deep learning framework for reconstructing 3D tree point clouds from Digital Surface Models (DSMs) and orthophoto imagery.
All the material can be seen in: [treeON](https://angelikigram.github.io/treeON/)

## Pretrained Model Weights

Pretrained model weights are available for download from Google Drive:

[Download pretrained model weights (Google Drive)](https://drive.google.com/file/d/1zVpkl4hREym_-UoGFSODrlAYETtT8Qcl/view?usp=sharing)

After downloading, place the extracted folder at: `log/treeON_model_weights/`

After downloading, place the folder `treeON_model_weights` inside the `log/` directory:

```
TreeON/
├── log/
│   └── treeON_model_weights/
```

To run the model using the pretrained weights, set the following argument:

```bash
--env treeON_model_weights
```

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

## Key Features

### 1. Multi-Modal Input
- **DSM (Digital Surface Model)**: 3D terrain height data
- **Orthophoto**: High-resolution aerial imagery
- **Combined Processing**: Leverages both geometric and visual information

### 2. Advanced Rendering Pipeline
The `gen_renderings.py` script provides:
- **Multi-Model Support**: Processes 19 different model variants automatically
- **Color Blending**: Intelligent mixing of point cloud colors with leaf textures
- **Histogram Matching**: Matches colors to reference orthophotos
- **Random Sampling**: Creates natural color variations
- **Texture Integration**: Applies realistic leaf and bark textures

### 3. Model Variants
The system supports multiple model configurations:
- **DSM-based**: `dsm_all`, `dsm_bce`, `dsm_shadow`, etc.
- **Mixed**: `mixed_all`, `mixed_bce_shadow`, `mixed_silhouettes`, etc.
- **Ortho-based**: `ortho_all`, `ortho_bce_shadow`, etc.

### 4. Output Formats
- **Point Clouds**: `.ply` files with color information
- **Renderings**: High-quality PNG images (512x512)
- **Metadata**: Color data and processing information in text format

## Key Parameters

- `--variable`: Use normalized query points
- `--num_trees`: Total number of trees in the dataset
- `--deciduous`: Include deciduous trees (true/false)
- `--shadow`: Enable shadow generation
- `--silhouettes`: Enable silhouette processing
- `--colors`: Enable color information
- `--model`: Model architecture variant (1-6)

## Output

Training outputs are saved in:
- `log/`: Training logs and metrics
- `val_npy/`: Validation results in NumPy format (for validation with training data)
- `results/TREE_MODELS`: Generated point clouds and renderings

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
    --env treeON_model_weights \
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

## Validation

Run validation pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py \
    --env treeON_model_weights \
    --num_query_points 85000 \
    --top_k 4000 \
    --num_points 4000 \
    --model 4 \
    --deciduous true \
    --variable 2 \
    --top_k_max 1200

CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py --env treeON_model_weights --num_query_points 85000 --top_k 4000 --num_points 4000 --model 1 --deciduous true --variable 3 --top_k_max 12000
```

## Dataset

The project uses Austrian landmarks dataset located in `landmarks_austria/` directory:

- `trees-data.csv`: Tree metadata
- `DATA_LANDMARKS/`: Main dataset directory
- `DSM/`: Digital Surface Models
- `ORTHOPHOTOS/`: Orthographic images

## Requirements

- **Python 3.8+**
- **PyTorch** with CUDA support
- **Blender 4.0+** (for rendering)
- **NumPy, scikit-image, PIL**
- **Visdom** (for visualization)
- **CUDA toolkit** (for extensions)

## Contributing

When adding new features:
1. Place core algorithms in `auxiliary/`
2. Add validation scripts to `validation/`
3. Update model configurations in the main training script
4. Document new output formats in this README