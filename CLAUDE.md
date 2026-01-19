# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAGA (Segment Any 3D Gaussians) is a 3D scene understanding system that combines 3D Gaussian Splatting with semantic segmentation capabilities. The project enables:
- Novel view synthesis via 3D Gaussian Splatting
- Interactive 3D segmentation and clustering
- Open-vocabulary segmentation using CLIP features
- Text-based segmentation using Grounding DINO

## Common Development Commands

### Installation
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

### Training Pipeline (Three Stages)

**1. Pre-train 3D Gaussians** (Standard 3DGS for scene reconstruction)
```bash
python train_scene.py -s <path to COLMAP or NeRF dataset>
```

**2. Generate Segmentation Masks** (Using SAM + Grounding DINO)
```bash
# Extract SAM masks and scale information
python extract_segment_everything_masks.py --image_root <path> --sam_checkpoint_path <path> --downsample <1/2/4/8>
python get_scale.py --image_root <path> --model_path <path to 3DGS model>

# Optional: For text-based segmentation, use Grounding DINO
python segment.py  # Uses hydra configs from configs/segment.yaml
```

**3. Train Affinity Features**
```bash
# Train contrastive instance features
python train_contrastive_feature.py -m <path to 3DGS model> --iterations 10000 --num_sampled_rays 1000

# Or train semantic features with labels
python train_feature.py  # Uses hydra configs from configs/training.yaml
```

### Segmentation and Rendering

**Interactive GUI** (for manual 3D segmentation and clustering)
```bash
python saga_gui.py --model_path <path to 3DGS model>
```

GUI Controls:
- Left drag: Rotate camera
- Mid drag: Pan camera
- Right click: Input point prompts for segmentation

**Render Segmented Scenes**
```bash
# Render scene with segmented object (remove background)
python render.py -m <model_path> --precomputed_mask <mask.pt> --target scene --segment

# Render 2D segmentation masks
python render.py -m <model_path> --precomputed_mask <mask.pt> --target seg

# Render original scene without segmentation
python render.py -m <model_path> --target scene
```

## Architecture Overview

### Core Gaussian Models

- **[scene/gaussian_model.py](scene/gaussian_model.py)**: `GaussianModel` - Base 3D Gaussian Splatting model with standard attributes (xyz, features_dc, features_rest, scaling, rotation, opacity)

- **[scene/feature_gaussian_model.py](scene/feature_gaussian_model.py)**: `FeatureGaussianModel` - Extends base model with:
  - `_instance_feature`: Contrastive features for instance segmentation (default dim=32)
  - `_semantic_feature`: Semantic features for class-specific segmentation (default dim=32)
  - Both features use L2 normalization activation

### Training Scripts

- **[train_scene.py](train_scene.py)**: Pre-trains base 3D Gaussians using standard photometric loss (L1 + SSIM). Inherited from original 3DGS implementation.

- **[train_contrastive_feature.py](train_contrastive_feature.py)**: Trains instance features using contrastive learning:
  - Uses SAM masks as pseudo-labels
  - Intra-mask loss: maximize similarity within same mask
  - Inter-mask loss: minimize similarity between different masks
  - Key hyperparameters: `ray_sample_rate`, `num_sampled_rays`, `rfn` (feature norm weight)

- **[train_feature.py](train_feature.py)**: Trains semantic features with class labels:
  - Uses both contrastive and semantic losses
  - Supports hydra configuration via configs/training.yaml
  - Requires pre-computed class label features

### Segmentation Pipeline

- **[segment.py](segment.py)**: Generates 2D segmentation masks:
  - Uses Grounding DINO for text-based object detection
  - Uses SAM for mask refinement
  - Configured via hydra (configs/segment.yaml)
  - Outputs: masks (.pt files) and class IDs

### Rendering

- **[gaussian_renderer/__init__.py](gaussian_renderer/__init__.py)**: Multiple render modes:
  - `render()`: Standard RGB rendering
  - `render_contrastive_feature()`: Instance feature rendering (L2 normalized)
  - `render_semantic_feature()`: Semantic feature rendering (L2 normalized)
  - `render_with_depth()`: Depth map rendering
  - `render_with_max_contributor()`: Returns which Gaussian contributes most to each pixel

### Interactive GUI

- **[saga_gui.py](saga_gui.py)**: DearPyGui-based interactive tool for:
  - Novel view synthesis
  - Point-prompt based 3D segmentation
  - 3D clustering using HDBSCAN
  - Visualization modes: RGB, PCA features, similarity maps, cluster colors

## Configuration System

The project uses **Hydra** for configuration management. Config files are in `configs/`:
- `configs/training.yaml`: Model, dataset, and training hyperparameters
- `configs/segment.yaml`: Segmentation settings (classes, thresholds, paths)
- `configs/model/`, `configs/dataset/`, `configs/pipe/`: Modular config components

Configuration composes via defaults, e.g.:
```yaml
defaults:
  - model: farsee
  - dataset: farsee
  - pipe: default
  - training: farsee
```

## Data Structure

Expected scene data layout:
```
data/<dataset_name>/<scene_name>/
  ├── images/          # Input images
  ├── images_2/        # Downsampled by 2x
  ├── images_4/        # Downsampled by 4x
  ├── images_8/        # Downsampled by 8x
  ├── sparse/          # COLMAP sparse reconstruction
  ├── sam_masks/       # Pre-computed SAM masks (.pt files)
  ├── mask_scales/     # Scale information for masks
  ├── features/        # CLIP features (optional, for open-vocab)
  └── depth/           # Depth maps (optional)
```

## Key Dependencies and Submodules

- **submodules/diff-gaussian-rasterization**: Custom CUDA rasterizer for 3D Gaussians
- **submodules/simple-knn**: K-NN for point cloud operations
- **third_party/segment-anything**: SAM model for segmentation
- **third_party/GroundingDINO**: Text-based object detection

## Important Implementation Details

1. **Feature Normalization**: Instance and semantic features are L2 normalized before rendering (see `feature_activation` in FeatureGaussianModel)

2. **Depth Reuse**: In [train_feature.py](train_feature.py:228-242), depth maps are computed once and reused for both feature renders to save memory

3. **Memory Management**: Training scripts use `torch.cuda.empty_cache()` strategically to handle GPU memory constraints

4. **Camera Conventions**: The codebase uses a specific coordinate convention where `c2w = [Rc|-Rc.T@tc]` (see saga_gui.py)

5. **Rotation**: Some workflows rotate images 90° clockwise for processing (see segment.py)
