import json
import math
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from utils.graphics_utils import fov2focal
from scene.dataset_readers import storePly, fetchPly
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB

from ..specs import CameraParams, CameraDataIndex, CameraSpec
from ..scene_index import SceneIndex, split_train_test, compute_nerfpp_normalization


class BlenderReader:
    def __init__(self, source_path: Path, extension: str = ".png"):
        self.source_path = Path(source_path)
        self.extension = extension

    def _read_transforms(self, transforms_file: str, start_idx: int = 0) -> List[CameraSpec]:
        transforms_path = self.source_path / transforms_file
        with open(transforms_path) as f:
            contents = json.load(f)

        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        specs = []
        for idx, frame in enumerate(frames):
            if transforms_file == "transforms_train.json":
                image_path = self.source_path / "train" / f"r_{idx+1:03d}{self.extension}"
            else:
                image_path = self.source_path / "test" / f"r_{idx+1:03d}{self.extension}"

            image = Image.open(image_path)
            width, height = image.size

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            fx = fov2focal(fovx, width)
            fovy = 2 * math.atan((height / width) * math.tan(fovx / 2))
            fy = fov2focal(fovy, height)

            params = CameraParams(
                uid=start_idx + idx,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=width / 2,
                cy=height / 2,
                R=R,
                T=T,
            )

            data_index = CameraDataIndex(image_path=image_path)
            specs.append(CameraSpec(params=params, data_index=data_index, image_name=image_path.stem))
        return specs

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneIndex:
        train_specs = self._read_transforms("transforms_train.json", start_idx=0)
        test_specs = self._read_transforms("transforms_test.json", start_idx=len(train_specs))

        specs = train_specs + test_specs
        scene_transform = compute_nerfpp_normalization(train_specs if train_specs else specs)

        ply_path = self.source_path / "points3d.ply"
        if not ply_path.exists():
            num_pts = 100_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(str(ply_path), xyz, SH2RGB(shs) * 255)
        else:
            pcd = fetchPly(str(ply_path))

        if eval:
            # Already split by transforms_train/test
            train_ids = list(range(len(train_specs)))
            test_ids = list(range(len(train_specs), len(specs)))
        else:
            train_ids = list(range(len(specs)))
            test_ids = []

        return SceneIndex(
            specs=specs,
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )
