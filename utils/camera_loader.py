"""
Camera data loading module.

This module provides the CameraLoader class which is responsible for loading
all camera-related data including images, masks, labels, depth maps, and confidence maps.
It handles resolution scaling, data validation, and error handling.
"""

from typing import Optional, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

from scene.camera_spec import CameraSpec
from scene.camera_data import CameraData
from utils.camera_utils import read_dmb_file
from utils.general_utils import PILtoTorch


class DataLoadError(Exception):
    """
    Exception raised when data loading fails.

    Attributes:
        path: Path to the file that failed to load
        reason: Human-readable reason for the failure
    """

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load {path}: {reason}")


class CameraLoader:
    """
    Camera data loader responsible for loading all camera-related data.

    This loader handles:
    - Image loading and resolution adjustment
    - Mask loading and interpolation
    - Label and feature loading
    - Depth map loading and scaling
    - Confidence map loading and scaling

    All data is properly validated and error messages are clear and actionable.

    Args:
        resolution_scale: Scale factor for resolution (1.0 = full resolution)
        resolution: Target resolution or one of [1, 2, 4, 8, -1]
                    -1 means auto-scale large images to 1600px width
    """

    def __init__(self, resolution_scale: float = 1.0, resolution: int = 1):
        self.resolution_scale = resolution_scale
        self.resolution = resolution

    def load(self, spec: CameraSpec) -> CameraData:
        """
        Load all camera data from a CameraSpec.

        Args:
            spec: CameraSpec containing metadata and file paths

        Returns:
            CameraData containing all loaded tensors

        Raises:
            DataLoadError: If any required file fails to load
        """
        data = CameraData()

        # 1. Load image
        data.image, data.alpha_mask = self._load_image(spec)

        # 2. Load masks
        if spec.mask_path:
            data.masks = self._load_masks(spec.mask_path, data.image.shape)

        # 3. Load labels
        if spec.labels_path:
            data.labels, data.label_features = self._load_labels(spec.labels_path)

        # 4. Load depth
        if spec.depth_path:
            data.depth_map = self._load_depth(spec.depth_path, data.image.shape)

        # 5. Load confidence
        if spec.confidence_path:
            data.confidence_map = self._load_confidence(spec.confidence_path, data.image.shape)

        data._loaded = True
        return data

    def _load_image(self, spec: CameraSpec) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load image and adjust resolution.

        Args:
            spec: CameraSpec with image_path

        Returns:
            Tuple of (image_tensor, alpha_mask_tensor)
            - image_tensor: (3, H, W) tensor in [0, 1] range
            - alpha_mask_tensor: (1, H, W) tensor or None

        Raises:
            DataLoadError: If image file cannot be loaded
        """
        try:
            image = Image.open(spec.image_path)
        except FileNotFoundError:
            raise DataLoadError(spec.image_path, "Image file not found")
        except Exception as e:
            raise DataLoadError(spec.image_path, f"Failed to load image: {str(e)}")

        # Calculate target resolution
        orig_w, orig_h = image.size
        resized_w, resized_h = self._calculate_target_resolution(orig_w, orig_h)

        # Convert to tensor
        resized_image_rgb = PILtoTorch(image, (resized_w, resized_h))
        gt_image = resized_image_rgb[:3, ...]
        gt_alpha_mask = None

        # Handle alpha channel
        if resized_image_rgb.shape[0] == 4:
            gt_alpha_mask = resized_image_rgb[3:4, ...]

        return gt_image, gt_alpha_mask

    def _calculate_target_resolution(self, orig_w: int, orig_h: int) -> Tuple[int, int]:
        """
        Calculate target resolution based on settings.

        Args:
            orig_w: Original image width
            orig_h: Original image height

        Returns:
            Tuple of (target_width, target_height)
        """
        if self.resolution in [1, 2, 4, 8]:
            scale = self.resolution * self.resolution_scale
        else:
            # Custom resolution or auto-scale
            if self.resolution == -1:
                # Auto-scale large images
                if orig_w > 1600:
                    global WARNED
                    if not hasattr(self, '_warned') or not self._warned:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), "
                              "rescaling to 1.6K.\n "
                              "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        self._warned = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / self.resolution
            scale = float(global_down) * float(self.resolution_scale)

        resized_w = int(orig_w / scale)
        resized_h = int(orig_h / scale)

        return resized_w, resized_h

    def _load_masks(self, mask_path: Path, image_shape: tuple) -> torch.Tensor:
        """
        Load masks and adjust to image resolution.

        Args:
            mask_path: Path to mask .pt file
            image_shape: Shape of the loaded image (C, H, W)

        Returns:
            Masks tensor of shape (N, H, W) with dtype bool

        Raises:
            DataLoadError: If mask file cannot be loaded
        """
        try:
            masks = torch.load(mask_path, weights_only=True)
        except FileNotFoundError:
            raise DataLoadError(mask_path, "Mask file not found")
        except Exception as e:
            raise DataLoadError(mask_path, f"Failed to load masks: {str(e)}")

        _, target_h, target_w = image_shape

        if masks.shape[0] == 0:
            # Empty mask
            return torch.empty((0, target_h, target_w), dtype=torch.bool)

        # Resize using bilinear interpolation
        masks_float = masks.float()
        resized_masks_float = F.interpolate(
            masks_float.unsqueeze(1),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        resized_masks = (resized_masks_float > 0.5).bool()

        return resized_masks

    def _load_labels(self, labels_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load labels and label features.

        Args:
            labels_path: Path to labels .pt file

        Returns:
            Tuple of (labels, label_features)
            - labels: (N,) tensor of int64 class IDs
            - label_features: (N, D) tensor of float32 features

        Raises:
            DataLoadError: If label files cannot be loaded
        """
        try:
            labels = torch.load(labels_path, weights_only=True)
        except Exception as e:
            raise DataLoadError(labels_path, f"Failed to load labels: {str(e)}")

        # Load label features
        label_features_path = labels_path.parent / 'label_features.pt'
        try:
            label_features = torch.load(label_features_path, weights_only=True)
        except Exception as e:
            raise DataLoadError(label_features_path, f"Failed to load label features: {str(e)}")

        return labels, label_features

    def _load_depth(self, depth_path: Path, image_shape: tuple) -> torch.Tensor:
        """
        Load depth map and adjust to image resolution.

        Args:
            depth_path: Path to depth .dmb file
            image_shape: Shape of the loaded image (C, H, W)

        Returns:
            Depth map tensor of shape (1, H, W)

        Raises:
            DataLoadError: If depth file cannot be loaded
        """
        try:
            depth_array = read_dmb_file(depth_path, is_confidence=False)
        except Exception as e:
            raise DataLoadError(depth_path, f"Failed to load depth: {str(e)}")

        _, target_h, target_w = image_shape

        depth_map = torch.from_numpy(depth_array.copy())[None, ...]
        resized_depth_map = F.interpolate(
            depth_map.unsqueeze(0),
            size=(target_h, target_w),
            mode='nearest',
        ).squeeze(0)

        return resized_depth_map

    def _load_confidence(self, conf_path: Path, image_shape: tuple) -> torch.Tensor:
        """
        Load confidence map and adjust to image resolution.

        Args:
            conf_path: Path to confidence .dmb file
            image_shape: Shape of the loaded image (C, H, W)

        Returns:
            Confidence map tensor of shape (1, H, W)

        Raises:
            DataLoadError: If confidence file cannot be loaded
        """
        try:
            conf_array = read_dmb_file(conf_path, is_confidence=True)
        except Exception as e:
            raise DataLoadError(conf_path, f"Failed to load confidence: {str(e)}")

        _, target_h, target_w = image_shape

        conf_map = torch.from_numpy(conf_array.copy())[None, ...]
        resized_conf_map = F.interpolate(
            conf_map.unsqueeze(0),
            size=(target_h, target_w),
            mode='nearest',
        ).squeeze(0)

        return resized_conf_map
