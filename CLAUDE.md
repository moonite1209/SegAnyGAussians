# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAGA (Segment Any 3D Gaussians) is a 3D scene understanding system that combines 3D Gaussian Splatting with semantic segmentation capabilities. The project enables:
- Novel view synthesis via 3D Gaussian Splatting
- Interactive 3D segmentation and clustering
- Text-based segmentation using Grounding DINO + SAM
- Semantic and instance feature learning for 3D Gaussians

**Note:** This project has some legacy files from historical development. The main active workflow scripts are:
- [train_scene.py](train_scene.py) - Pre-train 3D Gaussians for scene reconstruction
- [segment.py](segment.py) - Generate segmentation masks and labels from input images
- [train_feature.py](train_feature.py) - Train semantic and instance features
- [cluster.py](cluster.py) - Cluster Gaussians and assign semantic labels
- [gui.py](gui.py) - Interactive viewer for results

## Common Development Commands

### Installation
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

### Complete Pipeline

**1. Pre-train 3D Gaussians** (Standard 3DGS for scene reconstruction)
```bash
python train_scene.py -s <path to COLMAP or NeRF dataset>
```

**2. Generate Segmentation Masks** (Using Grounding DINO + SAM)
```bash
python segment.py  # Uses hydra configs from configs/segment.yaml
```
This generates:
- `masks/` - Binary segmentation masks (.pt files)
- `labels/` - Class label indices (.pt files)
- `label_features.pt` - Semantic feature vectors for each class

**3. Train Semantic and Instance Features**
```bash
python train_feature.py  # Uses hydra configs from configs/training.yaml
```
This trains both:
- Instance features for object-level discrimination
- Semantic features for class-level understanding

**4. Cluster and Assign Labels**
```bash
python cluster.py  # Uses hydra configs from configs/clustering.yaml
```
This performs:
- Hybrid clustering using instance features, semantic features, and spatial (xyz) information
- Class label assignment via semantic feature similarity
- Outputs to `cluster_output.json` containing point labels and instance-to-class mappings

**5. View Results**
```bash
python gui.py  # Uses hydra configs from configs/gui.yaml
```
Interactive viewer with:
- Render modes: RGB, instance feature (PCA), semantic feature (PCA), cluster colors
- Filters: by label, scale, opacity, weight
- Camera controls: left-drag to orbit, right-drag to pan, scroll to zoom

### Rendering

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
  - `_instance_feature`: Instance features for object-level discrimination (default dim=32)
  - `_semantic_feature`: Semantic features for class-level understanding (default dim=32)
  - Both features use L2 normalization activation

### Main Workflow Scripts

- **[train_scene.py](train_scene.py)**: Pre-trains base 3D Gaussians using standard photometric loss (L1 + SSIM). Inherited from original 3DGS implementation.

- **[segment.py](segment.py)**: Generates 2D segmentation masks using Grounding DINO + SAM:
  - Uses Grounding DINO for text-based object detection (configurable class list)
  - Uses SAM for high-quality mask refinement
  - Images are rotated 90° clockwise before processing
  - Configured via hydra (configs/segment.yaml)
  - Outputs: masks (.pt files), class IDs (.pt files), and label_features.pt

- **[train_feature.py](train_feature.py)**: Trains both semantic and instance features:
  - Uses 2D segmentation masks from segment.py
  - Contrastive loss: maximize similarity within same mask, minimize between different masks
  - Semantic loss: align semantic features with class label features
  - Key hyperparameters: `sample_rate`, `rfn` (feature norm weight), `instance_feature_lr`, `semantic_feature_lr`
  - Depth maps are computed once and reused for efficiency
  - Configured via hydra (configs/training.yaml)

- **[cluster.py](cluster.py)**: Performs 3D clustering and label assignment:
  - Hybrid clustering on (instance features, semantic features, xyz) with configurable ratios
  - Uses HDBSCAN for clustering (min_cluster_size=10, cluster_selection_epsilon=0.01)
  - Assigns class labels via semantic feature similarity (threshold=0.99)
  - Optional SOR (Statistical Outlier Removal) filtering for each cluster
  - Outputs `cluster_output.json` with point_labels and instances mappings
  - Configured via hydra (configs/clustering.yaml)

- **[gui.py](gui.py)**: DearPyGui-based interactive viewer:
  - Render modes: RGB, instance feature (PCA), semantic feature (PCA), cluster colors
  - Filters: by label, scale, opacity, weight
  - Loads clustering results from `cluster_output.json`
  - Camera controls: left-drag to orbit, right-drag to pan, scroll to zoom
  - Configured via hydra (configs/gui.yaml)

### Rendering

- **[gaussian_renderer/__init__.py](gaussian_renderer/__init__.py)**: Multiple render modes:
  - `render()`: Standard RGB rendering
  - `render_contrastive_feature()`: Instance feature rendering (L2 normalized)
  - `render_semantic_feature()`: Semantic feature rendering (L2 normalized)
  - `render_with_depth()`: Depth map rendering
  - `render_with_max_contributor()`: Returns which Gaussian contributes most to each pixel

### Legacy Scripts (Historical)

The following scripts are kept for reference but are not part of the main active workflow:
- [train_contrastive_feature.py](train_contrastive_feature.py) - Older contrastive feature training (superseded by train_feature.py)
- [saga_gui.py](saga_gui.py) - Legacy GUI (superseded by gui.py)
- [extract_segment_everything_masks.py](extract_segment_everything_masks.py), [get_scale.py](get_scale.py) - Older mask generation pipeline

## Configuration System

The project uses **Hydra** for configuration management. Main config files:
- `configs/training.yaml`: Entry point for training configuration
- `configs/segment.yaml`: Entry point for segmentation configuration
- `configs/clustering.yaml`: Entry point for clustering configuration
- `configs/gui.yaml`: Entry point for GUI configuration

Sub-config directories:
- `configs/model/`: Model parameters (sh_degree, feature dims, white_background)
- `configs/dataset/`: Dataset paths and parameters
- `configs/pipe/`: Pipeline parameters (NDC, render resolutions)
- `configs/training/`: Training hyperparameters (learning rates, densification, loss weights)
- `configs/segment/`: Segmentation settings (classes, thresholds, SAM/DINO paths)
- `configs/clustering/`: Clustering settings (feature ratios, thresholds, SOR parameters)
- `configs/gui/`: GUI settings (window size, paths)

Configuration composes via defaults, e.g.:
```yaml
defaults:
  - model: farsee
  - dataset: farsee
  - pipe: default
  - training: farsee  # or segment/clustering/gui
  - _self_

base_path: data/temp/1750728479386.658  # Override with actual path
```

## Data Structure

Expected scene data layout:
```
data/<dataset_name>/<scene_name>/
  ├── images/          # Input images
  ├── sparse/          # COLMAP sparse reconstruction
  ├── masks/           # Generated by segment.py (.pt files)
  ├── labels/          # Generated by segment.py (.pt files)
  └── output_models/   # 3DGS training output
      └── point_cloud/
          └── iteration_30000/
              └── point_cloud.ply
```

After segmentation:
```
data/<dataset_name>/<scene_name>/
  ├── masks/                    # 2D segmentation masks
  ├── labels/                   # Class label indices per mask
  ├── contrastive_feature_point_cloud.ply  # Instance features
  ├── feature_point_cloud.ply   # Semantic features
  └── cluster_output.json       # Clustering results
```

## Key Dependencies and Submodules

- **submodules/diff-gaussian-rasterization**: Custom CUDA rasterizer for 3D Gaussians
- **submodules/simple-knn**: K-NN for point cloud operations
- **third_party/segment-anything**: SAM model for segmentation
- **third_party/GroundingDINO**: Text-based object detection
- **weights/**: Download SAM and Grounding DINO checkpoints here

## Important Implementation Details

1. **Feature Normalization**: Instance and semantic features are L2 normalized before rendering (see `feature_activation` in FeatureGaussianModel)

2. **Depth Reuse**: In [train_feature.py](train_feature.py), depth maps are computed once and reused for both feature renders to save memory

3. **Memory Management**: Training scripts use `torch.cuda.empty_cache()` strategically to handle GPU memory constraints

4. **Camera Conventions**: The codebase uses a specific coordinate convention where `c2w = [Rc|-Rc.T@tc]` (see gui.py:OrbitCamera)

5. **Image Rotation**: [segment.py](segment.py:129) rotates images 90° clockwise before processing with Grounding DINO + SAM

6. **Label Features**: [segment.py](segment.py:90-111) generates deterministic semantic feature vectors for class names using hash-based seeding

7. **Clustering Ratios**: In cluster.py, the three feature ratios (instance_feature_ratio, semantic_feature_ratio, xyz_feature_ratio) must sum to 1.0
