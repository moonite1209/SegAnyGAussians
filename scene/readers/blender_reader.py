"""
Blender/NeRF synthetic scene reader.

Reads transforms_train.json and transforms_test.json files
and creates CameraSpec objects directly.
"""

import logging
import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from scene.scene_reader import SceneReader, SceneMetadata
from scene.camera_spec import CameraParams, CameraMetaData
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.graphics_utils import fov2focal

logger = logging.getLogger(__name__)


class BlenderReader(SceneReader):
    """
    Blender/NeRF synthetic dataset reader.

    Reads the standard NeRF synthetic dataset format with transforms JSON files.

    Args:
        source_path: Path to dataset root
        white_background: Whether to use white background
        extension: Image file extension (default: ".png")
    """

    def __init__(
        self,
        source_path: Path,
        white_background: bool = False,
        extension: str = ".png"
    ):
        self.source_path = Path(source_path)
        self.white_background = white_background
        self.extension = extension

    def _read_transforms(self, transforms_file: str, start_idx: int = 0) -> List:
        """
        Read transforms JSON file and create (CameraParams, CameraMetaData) list.

        Args:
            transforms_file: Name of transforms JSON file
            start_idx: Starting index for camera IDs

        Returns:
            List of (CameraParams, CameraMetaData) tuples
        """
        transforms_path = self.source_path / transforms_file

        with open(transforms_path) as f:
            contents = json.load(f)

        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        specs = []
        for idx, frame in enumerate(frames):
            # Load image to get dimensions
            image_rel_path = Path(frame["file_path"])
            if transforms_file == "transforms_train.json":
                image_path = self.source_path / "train" / f"r_{idx+1:03d}{self.extension}"
            else:
                image_path = self.source_path / "test" / f"r_{idx+1:03d}{self.extension}"

            image = Image.open(image_path)

            # Extract camera parameters
            width, height = image.size

            # Extract pose (convert from Blender to COLMAP convention)
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            # Compute focal lengths from FoV
            from utils.graphics_utils import focal2fov
            fx = fov2focal(fovx, width)
            fovy = focal2fov(fov2focal(fovx, width), height)
            fy = fov2focal(fovy, height)

            # Create CameraParams (rendering parameters)
            params = CameraParams(
                uid=start_idx + idx,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=width / 2,  # Blender datasets assume centered principal point
                cy=height / 2,
                R=R,
                T=T,
            )

            # Create CameraMetaData (data file paths)
            metadata = CameraMetaData(
                image_name=image_rel_path.stem,
                image_path=image_path,
                mask_path=None,
                labels_path=None,
                depth_path=None,
                confidence_path=None
            )

            specs.append((params, metadata))

        return specs

    def read_camera_specs(self) -> List:
        """
        Read all camera specs (train + test).

        Returns:
            Combined list of train and test (CameraParams, CameraMetaData) tuples
        """
        train_specs = self._read_transforms("transforms_train.json", start_idx=0)
        test_specs = self._read_transforms("transforms_test.json", start_idx=len(train_specs))

        return train_specs + test_specs

    def read_scene_metadata(self) -> SceneMetadata:
        """
        Read or generate synthetic scene metadata.

        Returns:
            SceneMetadata with point cloud and empty normalization
        """
        ply_path = self.source_path / "points3d.ply"

        if not ply_path.exists():
            # Generate random point cloud
            num_pts = 100_000
            logger.info(f"Generating random point cloud ({num_pts})...")

            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz,
                colors=SH2RGB(shs),
                normals=np.zeros((num_pts, 3))
            )

            from scene.dataset_readers import storePly
            storePly(str(ply_path), xyz, SH2RGB(shs) * 255)
            logger.info(f"Point cloud saved to {ply_path}")
        else:
            from scene.dataset_readers import fetchPly
            pcd = fetchPly(str(ply_path))

        return SceneMetadata(
            point_cloud=pcd,
            nerf_normalization={},  # To be filled by caller
            ply_path=ply_path
        )
