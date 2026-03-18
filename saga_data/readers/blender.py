from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from scene.dataset_readers import storePly
from utils.graphics_utils import fov2focal
from utils.sh_utils import SH2RGB

from ..scene_index import AssetRef, FrameRecord, SceneManifest, compute_nerfpp_normalization
from ..specs import CameraParams


class BlenderReader:
    def __init__(self, source_path: Path, extension: str = ".png"):
        self.source_path = Path(source_path)
        self.extension = extension

    def _read_transforms(self, transforms_file: str, start_idx: int = 0) -> list[FrameRecord]:
        transforms_path = self.source_path / transforms_file
        with open(transforms_path, encoding="utf-8") as handle:
            contents = json.load(handle)

        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        records: list[FrameRecord] = []
        for idx, frame in enumerate(frames):
            if transforms_file == "transforms_train.json":
                image_path = self.source_path / "train" / f"r_{idx + 1:03d}{self.extension}"
            else:
                image_path = self.source_path / "test" / f"r_{idx + 1:03d}{self.extension}"

            image = Image.open(image_path)
            width, height = image.size
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            r_matrix = -np.transpose(matrix[:3, :3])
            r_matrix[:, 0] = -r_matrix[:, 0]
            t_vector = -matrix[:3, 3]

            fx = fov2focal(fovx, width)
            fovy = 2 * math.atan((height / width) * math.tan(fovx / 2))
            fy = fov2focal(fovy, height)
            params = CameraParams(
                uid=start_idx + idx,
                width=width,
                height=height,
                fx=float(fx),
                fy=float(fy),
                cx=width / 2,
                cy=height / 2,
                R=r_matrix,
                T=t_vector,
            )
            records.append(
                FrameRecord(
                    frame_id=image_path.relative_to(self.source_path).with_suffix("").as_posix().replace("/", "__"),
                    image_name=image_path.stem,
                    params=params,
                    assets={
                        "image": AssetRef(
                            kind="image",
                            path=image_path,
                            native_shape=(height, width),
                            dtype="uint8",
                            codec=image_path.suffix,
                        )
                    },
                )
            )
        return records

    def read_scene_manifest(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        del llffhold
        train_frames = self._read_transforms("transforms_train.json", start_idx=0)
        test_frames = self._read_transforms("transforms_test.json", start_idx=len(train_frames))

        frames = train_frames + test_frames
        scene_transform = compute_nerfpp_normalization(train_frames if train_frames else frames)

        ply_path = self.source_path / "points3d.ply"
        if not ply_path.exists():
            num_pts = 100_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            storePly(str(ply_path), xyz, SH2RGB(shs) * 255)

        if eval:
            train_ids = list(range(len(train_frames)))
            test_ids = list(range(len(train_frames), len(frames)))
        else:
            train_ids = list(range(len(frames)))
            test_ids = []

        return SceneManifest(
            frames=tuple(frames),
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        return self.read_scene_manifest(eval=eval, llffhold=llffhold)
