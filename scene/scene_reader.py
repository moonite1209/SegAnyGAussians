"""
Unified scene reading interface.

This module provides the SceneReader abstract base class and related utilities
for reading scene data from various sources (COLMAP, Blender, LERF, etc.).

Design principles:
- Direct creation of CameraParams and CameraMetaData (bypass CameraInfo)
- Pluggable architecture (easy to add new readers)
- Consistent interface across all data types
- Separation of concerns (reading vs processing)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from scene.camera_spec import CameraParams, CameraMetaData
from scene.gaussian_model import BasicPointCloud


@dataclass
class SceneMetadata:
    """
    Scene-level metadata returned by readers.

    Attributes:
        point_cloud: Initial point cloud for Gaussian initialization
        nerf_normalization: Normalization dict with 'translate' and 'radius' keys
        ply_path: Path to point cloud .ply file
    """
    point_cloud: Optional[BasicPointCloud]
    nerf_normalization: dict
    ply_path: Path


class SceneReader(ABC):
    """
    Abstract base class for scene readers.

    All concrete readers must implement the read_camera_specs and
    read_scene_metadata methods. This class also provides utility
    methods for train/test splitting and normalization computation.

    Example:
        >>> reader = ColmapReader(sparse_path=..., images_path=...)
        >>> specs = reader.read_camera_specs()  # Returns list of (params, metadata) tuples
        >>> train_specs, test_specs = reader.split_train_test(specs, eval=True)
        >>> metadata = reader.read_scene_metadata()
    """

    @abstractmethod
    def read_camera_specs(self) -> List[Tuple[CameraParams, CameraMetaData]]:
        """
        Read all camera specifications from the data source.

        This method should create CameraParams and CameraMetaData objects
        directly from the source data, without going through CameraInfo.

        Returns:
            List of (CameraParams, CameraMetaData) tuples (created directly, not from CameraInfo)
        """
        pass

    @abstractmethod
    def read_scene_metadata(self) -> SceneMetadata:
        """
        Read scene-level metadata.

        Returns:
            SceneMetadata object containing point cloud and normalization info
        """
        pass

    def split_train_test(
        self,
        specs: List[Tuple[CameraParams, CameraMetaData]],
        eval: bool = False,
        llffhold: int = 8
    ) -> Tuple[List[Tuple[CameraParams, CameraMetaData]], List[Tuple[CameraParams, CameraMetaData]]]:
        """
        Split camera specs into train and test sets.

        Uses LLFF-style evaluation split: every llffhold-th image goes to test.

        Args:
            specs: List of (CameraParams, CameraMetaData) tuples
            eval: If True, split into train/test; if False, all go to train
            llffhold: Holdout period (every llffth image goes to test)

        Returns:
            Tuple of (train_specs, test_specs) as lists of tuples
        """
        if eval:
            train_specs = [s for idx, s in enumerate(specs) if idx % llffhold != 0]
            test_specs = [s for idx, s in enumerate(specs) if idx % llffhold == 0]
        else:
            train_specs = specs
            test_specs = []

        return train_specs, test_specs

    @staticmethod
    def get_nerfpp_normalization(specs: List[Tuple[CameraParams, CameraMetaData]]) -> dict:
        """
        Compute NeRF-style normalization from camera specs.

        This computes the center and radius of all camera centers for
        normalization purposes, following the NeRF++ paper convention.

        Args:
            specs: List of (CameraParams, CameraMetaData) tuples

        Returns:
            Dict with 'translate' (center) and 'radius' keys
        """
        from utils.graphics_utils import getWorld2View2

        cam_centers = []
        for params, _ in specs:
            W2C = getWorld2View2(params.R, params.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center  # Keep (3, 1) shape for broadcasting
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)

        radius = diagonal * 1.1
        translate = -center.flatten()  # Flatten only when creating translate

        return {"translate": translate, "radius": radius}
