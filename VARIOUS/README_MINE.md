# TreeON: 3D Tree Reconstruction from DSM and Orthophoto Data

This repository contains the complete implementation of TreeON, a deep learning framework for generating 3D tree point clouds from Digital Surface Models (DSM) and orthophoto images.

## Project Structure

### Core Training Files
- **`train.py`** - Main training script for the neural network models
- **`crop_images.py`** - Image preprocessing utility for cropping and background removal

### Auxiliary Modules (`auxiliary/`)
Core utilities and helper functions for the training pipeline:

- **`load_model.py`** - Model loading and checkpoint management
- **`loss_functions.py`** - Custom loss functions for 3D point cloud generation
- **`occupancy_compute.py`** - Occupancy field computation and point extraction
- **`query_points_gen.py`** - Query point generation for occupancy prediction
- **`shadow_gen.py`** - Shadow generation utilities
- **`utils.py`** - General utility functions
- **`visualize.py`** - Visualization tools for 3D point clouds

#### Sub-modules:
- **`dataset/`** - Dataset loading and preprocessing classes
- **`models/`** - Neural network model architectures

### Extensions (`extension/`)
CUDA extensions for efficient 3D operations:

- **`chamfer_cuda.cpp`** - C++ implementation of Chamfer distance
- **`chamfer.cu`** - CUDA kernel for Chamfer distance computation
- **`dist_chamfer.py`** - Python wrapper for Chamfer distance
- **`setup.py`** - Build script for CUDA extensions
- **`build/`** - Compiled extension binaries
- **`chamfer.egg-info/`** - Package metadata

### France Dataset (`france_data/`)
French dataset for validation:

- **`DSM_MAT/`** - Digital Surface Model files in MATLAB format:
  - `dsm_1.mat`, `dsm_2.mat`, `dsm_3.mat`, `dsm_4.mat`
- **`ORTHOPHOTOS/`** - Corresponding orthophoto images:
  - `ortho_1.png`, `ortho_2.png`, `ortho_3.png`, `ortho_4.png`

### Austrian Landmarks Dataset (`landmarks_austria/`)
Main dataset with Austrian terrain and tree data:

- **`convert_mat_to_obj.py`** - Utility to convert MATLAB files to OBJ format
- **`trees-data.csv`** - Metadata file with tree categories and properties

#### Data Folders:
- **`DATA_LANDMARKS/`** - Raw landmark data:
  - `DSM/` - Digital Surface Models (cut as the dataset P2/ --- DSM_1 not cut as ./ dataset in the server)
  - `DSM_OBJ/` - DSM data in OBJ format
  - `DSM_OBJ/` - DSM data in OBJ format
  - `DSM_OBJ_NORMALIZED_ALIGNED/` - DSM data in the format we need 
To generate test data: run under gen_test_trees/ and also align_dsm.py and then convert_obj_to_mat.py
  - `ORTHOPHOTOS/` - Orthophoto images
  - Various processing subdirectories
  - `DATA-GEN/` - To generate test data (trees, dsm, orthophotos)
    - `ORTHOPHOTOS/` - Terrain patches data 
- **`TREE_MODELS/`** - Generated 3D tree models organized by model type

### Validation (`validation/`)
Testing and validation pipeline:

- **`gen_renderings.py`** - **Main rendering script** - Generates high-quality tree renderings from point clouds using Blender
- **`validation_pipeline.py`** - Complete validation workflow
- **`validation_pipeline_landmarks.py`** - Landmark-specific validation
- **`color_metrics.py`** - Color accuracy evaluation metrics

#### Sub-modules:
- **`gen_point_cloud/`** - Point cloud generation from trained models:
  - `generate_output_pointcloud_from_files.py` - Main point cloud generation script
- **`Scenes_Gen/`** - Scene generation utilities (currently in GEN_SCENES)
- **`various/`** - Additional validation tools

### Output and Results (`results/`, `log/`, `val_npy/`)

- **`results/`** - Main results directory:
  - `outputs/{MODEL_NAME}/` - Model-specific output folders containing:
    - `{tree_id}.png` - Final rendered images
    - `{tree_id}_blended_colors.png` - Images with color blending applied
    - `{tree_id}_histogram_reference.png` - Reference textures used
    - `{tree_id}_leaf_colors.txt` - Extracted leaf texture colors
    - `{tree_id}_blended_colors.txt` - Final blended color data
  - `temp/` - Temporary processing files
  - `textures/` - Texture files for rendering

- **`log/`** - Training logs and checkpoints
- **`val_npy/`** - Validation data in NumPy format

### Miscellaneous
- **`VARIOUS/`** - Additional utilities and experimental code
- **`.gitignore`** - Git ignore rules
- **`__README_VAL.md`** - Validation-specific documentation

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

## Usage

### Training
```bash
python train.py --model 1 --num_points 4500 --env training_env
```

### Point Cloud Generation
```bash
python validation/gen_point_cloud/generate_output_pointcloud_from_files.py \
  --dsm_mat france_data/DSM_MAT/dsm_1.mat \
  --ortho_img france_data/ORTHOPHOTOS/ortho_1.png \
  --output_root results/france \
  --model_path checkpoints/model.pth
```

### Rendering (All Models)
```bash
# Using Blender (recommended)
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python validation/gen_renderings.py
```

### Image Preprocessing
```bash
python crop_images.py  # Configure paths in the script
```

## Requirements

- **Python 3.8+**
- **PyTorch** with CUDA support
- **Blender 4.0+** (for rendering)
- **NumPy, scikit-image, PIL**
- **Visdom** (for visualization)
- **CUDA toolkit** (for extensions)

## Key Algorithms

1. **Occupancy Prediction**: Neural network predicts 3D occupancy from 2D inputs
2. **Point Extraction**: Top-k occupied points selected for final point cloud
3. **Color Blending**: Sophisticated color mixing using:
   - 60-85% original point colors
   - 15-40% leaf texture colors
   - 75% orthophoto influence in histogram matching
4. **Random Sampling**: Natural color variation through stochastic pixel sampling

## Model Performance

The system generates realistic 3D tree reconstructions with:
- **Geometric Accuracy**: Faithful DSM-based shape reconstruction
- **Visual Realism**: Photorealistic color application
- **Scalability**: Processes multiple tree species and terrain types

## Contributing

When adding new features:
1. Place core algorithms in `auxiliary/`
2. Add validation scripts to `validation/`
3. Update model configurations in the main training script
4. Document new output formats in this README

For more specific documentation, see:
- `__README_VAL.md` - Validation pipeline details
- Individual script docstrings for function-level documentation