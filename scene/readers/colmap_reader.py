"""
COLMAP scene reader.

Reads COLMAP format scenes directly and creates CameraSpec objects
without going through CameraInfo.

Supported formats:
- Binary: images.bin, cameras.bin
- Text: images.txt, cameras.txt
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from scene.scene_reader import SceneReader, SceneMetadata
from scene.camera_spec import CameraParams, CameraMetaData
from scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text,
    qvec2rotmat
)
from scene.dataset_readers import fetchPly, storePly

logger = logging.getLogger(__name__)


class ColmapReader(SceneReader):
    """
    COLMAP format scene reader.

    Reads COLMAP binary/text files and creates CameraSpec objects directly.

    Args:
        sparse_path: Path to COLMAP sparse directory (contains images.bin/txt,
                     cameras.bin/txt, points3D.bin/txt/ply)
        images_path: Path to images directory
        masks_path: Optional path to masks directory (.pt files)
        labels_path: Optional path to labels directory (.pt files)
        depth_path: Optional path to depth directory (.dmb files)
    """

    def __init__(
        self,
        sparse_path: Path,
        images_path: Path,
        masks_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        depth_path: Optional[Path] = None
    ):
        self.sparse_path = Path(sparse_path)
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path) if masks_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.depth_path = Path(depth_path) if depth_path else None

        # Read COLMAP data (try binary, fallback to text)
        self.cam_extrinsics = self._read_extrinsics()
        self.cam_intrinsics = self._read_intrinsics()

    def _read_extrinsics(self):
        """Read camera extrinsics (try binary, fallback to text)."""
        try:
            cameras_extrinsic_file = self.sparse_path / "images.bin"
            return read_extrinsics_binary(str(cameras_extrinsic_file))
        except Exception:
            cameras_extrinsic_file = self.sparse_path / "images.txt"
            return read_extrinsics_text(str(cameras_extrinsic_file))

    def _read_intrinsics(self):
        """Read camera intrinsics (try binary, fallback to text)."""
        try:
            cameras_intrinsic_file = self.sparse_path / "cameras.bin"
            return read_intrinsics_binary(str(cameras_intrinsic_file))
        except Exception:
            cameras_intrinsic_file = self.sparse_path / "cameras.txt"
            return read_intrinsics_text(str(cameras_intrinsic_file))

    def _create_camera_spec(self, extr, intr, idx: int):
        """
        Create CameraParams and CameraMetaData directly from COLMAP data.

        This bypasses CameraInfo entirely and creates separate objects:
        - CameraParams: rendering parameters (fx, fy in pixels, not FoV)
        - CameraMetaData: data file paths

        Args:
            extr: COLMAP extrinsics data
            intr: COLMAP intrinsics data
            idx: Camera index

        Returns:
            Tuple of (CameraParams, CameraMetaData)
        """
        # Extract basic info
        uid = intr.id
        height = intr.height
        width = intr.width

        # Rotation and translation
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # Extract intrinsics based on camera model
        # NOTE: We extract fx, fy directly (in pixels), not FoV
        # This avoids the FoV ↔ focal conversion
        if intr.model == "SIMPLE_PINHOLE":
            fx = intr.params[0]  # Focal length in pixels
            fy = intr.params[0]  # Same for simple pinhole
            cx = intr.params[1]  # Principal point X
            cy = intr.params[2]  # Principal point Y
        elif intr.model == "PINHOLE":
            fx = intr.params[0]  # Focal length X
            fy = intr.params[1]  # Focal length Y
            cx = intr.params[2]  # Principal point X
            cy = intr.params[3]  # Principal point Y
        elif intr.model == "SIMPLE_RADIAL":
            fx = intr.params[0]
            fy = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        else:
            raise ValueError(
                f"COLMAP camera model {intr.model} not handled. "
                "Only undistorted datasets (PINHOLE or SIMPLE_PINHOLE) are supported."
            )

        # Create CameraParams (rendering parameters)
        params = CameraParams(
            uid=uid,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            R=R,
            T=T,
        )

        # Image name and paths
        image_name_noext = Path(extr.name).stem
        image_path = self.images_path / extr.name

        # Optional paths
        mask_path = self.masks_path / f"{image_name_noext}.pt" if self.masks_path else None
        labels_path = self.labels_path / f"{image_name_noext}.pt" if self.labels_path else None

        if self.depth_path:
            # LERF-style depth naming: {split}-{image_idx}_smoothDepth.dmb
            # Example: train-00000_smoothDepth.dmb
            image_idx = image_name_noext.split('-')[-1]
            depth_path = self.depth_path / f"{image_idx}_smoothDepth.dmb"
            confidence_path = self.depth_path / f"{image_idx}_confidence.dmb"
        else:
            depth_path = None
            confidence_path = None

        # Create CameraMetaData (data file paths)
        metadata = CameraMetaData(
            image_name=image_name_noext,
            image_path=image_path,
            mask_path=mask_path,
            labels_path=labels_path,
            depth_path=depth_path,
            confidence_path=confidence_path
        )

        return params, metadata

    def read_camera_specs(self) -> List:
        """
        Read all camera specs from COLMAP data.

        Returns:
            List of (CameraParams, CameraMetaData) tuples sorted by image name
        """
        cam_specs = []

        # Use tqdm for progress bar
        for key in tqdm(self.cam_extrinsics, desc="Reading cameras", unit="cam", leave=False):
            extr = self.cam_extrinsics[key]
            intr = self.cam_intrinsics[extr.camera_id]

            params, metadata = self._create_camera_spec(extr, intr, len(cam_specs))
            cam_specs.append((params, metadata))

        # Sort by image name for consistency
        cam_specs.sort(key=lambda s: s[1].image_name)

        logger.info(f"Loaded {len(cam_specs)} cameras from COLMAP")

        return cam_specs

    def read_scene_metadata(self) -> SceneMetadata:
        """
        Read scene metadata (point cloud).

        Returns:
            SceneMetadata with point cloud and placeholder normalization
        """
        # Read or generate point cloud
        ply_path = self.sparse_path / "points3D.ply"
        bin_path = self.sparse_path / "points3D.bin"
        txt_path = self.sparse_path / "points3D.txt"

        if not ply_path.exists():
            logger.info("Converting points3D.bin to .ply (first time only)...")
            try:
                xyz, rgb, _ = read_points3D_binary(str(bin_path))
            except Exception:
                xyz, rgb, _ = read_points3D_text(str(txt_path))
            storePly(str(ply_path), xyz, rgb)
            logger.info(f"Point cloud saved to {ply_path}")

        try:
            point_cloud = fetchPly(str(ply_path))
        except Exception:
            point_cloud = None

        # Normalization will be computed after we have camera specs
        # Return empty dict for now
        return SceneMetadata(
            point_cloud=point_cloud,
            nerf_normalization={},  # To be filled by caller
            ply_path=ply_path
        )
