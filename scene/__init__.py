#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Enhanced Scene classes using unified SceneReader architecture.

Key changes:
- Uses SceneReader interface for data loading
- Direct CameraSpec creation (no CameraInfo bottleneck)
- New Camera class with lazy loading
- Cleaner, more maintainable code
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Optional, Any
import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from scene.feature_gaussian_model import FeatureGaussianModel
from scene.camera import TrainCamera, Camera
from scene.camera_spec import CameraParams, CameraMetaData, from_camera_info
from scene.camera_dataset import CameraDataset
from scene.scene_reader import SceneReader, SceneMetadata
from scene.readers.colmap_reader import ColmapReader
from scene.readers.blender_reader import BlenderReader

logger = logging.getLogger(__name__)


class Scene:
    """
    Enhanced Scene class using unified reader architecture.

    Design principles:
    - Single data path: SceneReader → CameraSpec → Camera
    - Lazy loading: Camera data loaded on-demand
    - Memory efficient: Store only specs, not full camera data
    - Clean architecture: No CameraInfo intermediate step
    """

    gaussians: GaussianModel
    feature_gaussians: FeatureGaussianModel

    def __init__(
        self,
        args: Any,  # ModelParams-like object with path attributes
        gaussians: Optional[GaussianModel] = None,
        shuffle: bool = True,
        resolution_scales: list = [1.0],
        sample_rate: float = 1.0
    ):
        """
        Initialize Scene from various data sources.

        Args:
            args: Model parameters containing paths and settings
            gaussians: Optional pre-initialized Gaussian model
            shuffle: Whether to shuffle camera order
            resolution_scales: List of resolution scales to support
            sample_rate: Sampling rate for feature training
        """
        logger.info("Initializing Scene")

        self.model_path = args.model_path
        self.gaussians = gaussians

        self.train_specs = {}
        self.test_specs = {}
        self.train_cameras = {}
        self.test_cameras = {}

        # Create appropriate reader based on data source
        reader = self._create_reader(args, sample_rate)

        # Read camera specs and metadata
        all_specs = reader.read_camera_specs()
        metadata = reader.read_scene_metadata()

        # Compute normalization
        all_specs_sorted = sorted(all_specs, key=lambda s: s[1].image_name)
        nerf_norm = reader.get_nerfpp_normalization(all_specs_sorted)
        metadata.nerf_normalization = nerf_norm

        # Shuffle if requested
        if shuffle:
            random.shuffle(all_specs_sorted)

        # Split train/test
        train_specs, test_specs = reader.split_train_test(
            all_specs_sorted,
            eval=args.eval
        )

        # For compatibility, also store as tuples
        self.train_camera_tuples = train_specs
        self.test_camera_tuples = test_specs

        # Store specs (lightweight metadata)
        self.train_specs[1.0] = train_specs
        self.test_specs[1.0] = test_specs
        self.cameras_extent = nerf_norm["radius"]

        # Copy input.ply if needed
        if not os.path.exists(os.path.join(self.model_path, "input.ply")):
            with open(metadata.ply_path, 'rb') as src_file, \
                 open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())

        # Save cameras.json for compatibility
        if not os.path.exists(os.path.join(self.model_path, "cameras.json")):
            self._save_cameras_json(train_specs + test_specs)

        # Create cameras at each resolution scale
        for resolution_scale in resolution_scales:
            logger.info(f"Loading training cameras (scale={resolution_scale})")
            self.train_cameras[resolution_scale] = self._create_cameras(
                train_specs, resolution_scale, args
            )
            logger.info(f"Loaded {len(self.train_cameras[resolution_scale])} training cameras")

            logger.info(f"Loading test cameras (scale={resolution_scale})")
            self.test_cameras[resolution_scale] = self._create_cameras(
                test_specs, resolution_scale, args
            )
            logger.info(f"Loaded {len(self.test_cameras[resolution_scale])} test cameras")

        logger.info(f"Scene initialization complete: {len(train_specs)} train, {len(test_specs)} test")

    def _create_reader(
        self,
        args: Any,  # ModelParams-like object
        sample_rate: float
    ) -> SceneReader:
        """
        Create appropriate SceneReader based on data type.

        Args:
            args: Model parameters with path information
            sample_rate: Sampling rate

        Returns:
            SceneReader instance
        """
        if os.path.exists(os.path.join(args.sparse_path)):
            logger.info("Detected COLMAP dataset")
            return ColmapReader(
                sparse_path=args.sparse_path,
                images_path=args.images,
                masks_path=getattr(args, 'masks_path', None),
                labels_path=getattr(args, 'labels_path', None),
                depth_path=getattr(args, 'depth_path', None)
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            logger.info("Detected Blender/NeRF synthetic dataset")
            return BlenderReader(
                source_path=args.source_path,
                white_background=args.white_background
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            # LERF dataset - can use ColmapReader if it has sparse data
            # For now, treat as error until LERF reader is implemented
            raise ValueError(
                "LERF dataset detected. Please use COLMAP format or implement LERFReader."
            )
        else:
            raise ValueError("Could not recognize scene type!")

    def _create_cameras(
        self,
        camera_tuples: list,
        resolution_scale: float,
        args: Any  # ModelParams-like object
    ) -> list:
        """
        Create TrainCamera objects from (CameraParams, CameraMetaData) tuples.

        Args:
            camera_tuples: List of (CameraParams, CameraMetaData) tuples
            resolution_scale: Resolution scaling factor
            args: Model parameters

        Returns:
            List of TrainCamera objects with lazy loading enabled
        """
        cameras = []
        for params, metadata in camera_tuples:
            camera = TrainCamera(
                params=params,
                metadata=metadata,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0
            )
            cameras.append(camera)
        return cameras

    def _save_cameras_json(self, camera_tuples: list):
        """
        Save cameras.json for compatibility (legacy format).

        Args:
            camera_tuples: List of (CameraParams, CameraMetaData) tuples
        """
        json_cams = []
        for idx, (params, metadata) in enumerate(camera_tuples):
            # Create minimal camera info for JSON export
            from utils.graphics_utils import focal2fov

            cam_info = {
                "id": idx,
                "img_name": metadata.image_name,
                "width": params.width,
                "height": params.height,
                "fx": params.fx,
                "fy": params.fy,
                "cx": params.cx,
                "cy": params.cy,
                "depth_f": params.fx,  # Approximate
                "camera_angle_x": 2 * np.arctan(params.width / (2 * params.fx)),
                "camera_angle_y": 2 * np.arctan(params.height / (2 * params.fy)),
            }

            # Add pose information
            w2c = np.eye(4)
            w2c[:3, :3] = params.R
            w2c[:3, 3] = params.T
            cam_info["position"] = (-w2c[:3, :3].T @ w2c[:3, 3]).tolist()
            cam_info["rotation"] = w2c[:3, :3].T.tolist()

            json_cams.append(cam_info)

        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file, indent=2)

    def getTrainCameras(self, scale=1.0):
        """Get training cameras for specified scale."""
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """Get test cameras for specified scale."""
        return self.test_cameras[scale]


class FeatureScene:
    """
    Enhanced FeatureScene using new architecture.

    Uses CameraDataset for efficient batch loading during feature training.
    """

    def __init__(self, args):
        """
        Initialize FeatureScene.

        Args:
            args: Arguments containing paths and parameters
        """
        self.args = args

        # Create reader
        if os.path.exists(os.path.join(args.sparse_path)):
            logger.info("FeatureScene: Using COLMAP dataset")
            reader = ColmapReader(
                sparse_path=args.sparse_path,
                images_path=args.images_path,
                masks_path=args.masks_path,
                labels_path=args.labels_path,
                depth_path=args.depth_path
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            logger.info("FeatureScene: Using Blender dataset")
            reader = BlenderReader(
                source_path=args.source_path,
                white_background=args.white_background
            )
        else:
            raise ValueError("FeatureScene: Could not recognize scene type!")

        # Read specs
        all_specs = reader.read_camera_specs()
        all_specs_sorted = sorted(all_specs, key=lambda s: s[1].image_name)

        metadata = reader.read_scene_metadata()
        nerf_norm = reader.get_nerfpp_normalization(all_specs_sorted)

        # Split
        train_specs, test_specs = reader.split_train_test(
            all_specs_sorted,
            eval=args.eval
        )

        self.ply_path = metadata.ply_path
        self.train_specs = train_specs
        self.test_specs = test_specs
        self.cameras_extent = nerf_norm["radius"]

    def getCameraDataset(self, scale=1.0):
        """
        Get combined train+test dataset.

        Args:
            scale: Resolution scale factor

        Returns:
            CameraDataset with all cameras
        """
        all_specs = self.train_specs + self.test_specs
        return CameraDataset(
            specs=all_specs,
            resolution_scale=scale,
            resolution=self.args.resolution
        )

    def getTrainDataset(self, scale=1.0):
        """
        Get training dataset only.

        Args:
            scale: Resolution scale factor

        Returns:
            CameraDataset with training cameras
        """
        return CameraDataset(
            specs=self.train_specs,
            resolution_scale=scale,
            resolution=self.args.resolution
        )

    def getTestDataset(self, scale=1.0):
        """
        Get test dataset only.

        Args:
            scale: Resolution scale factor

        Returns:
            CameraDataset with test cameras
        """
        return CameraDataset(
            specs=self.test_specs,
            resolution_scale=scale,
            resolution=self.args.resolution
        )
