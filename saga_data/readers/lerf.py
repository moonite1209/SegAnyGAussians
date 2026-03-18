from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scene.dataset_readers import storePly
from utils.sh_utils import SH2RGB

from ..scene_index import AssetRef, FrameRecord, SceneManifest, compute_nerfpp_normalization, split_train_test
from ..specs import CameraParams


class LerfReader:
    def __init__(self, source_path: Path, extension: str = ".jpg"):
        self.source_path = Path(source_path)
        self.extension = extension

    def _read_transforms(self, transforms_file: str) -> list[FrameRecord]:
        transforms_path = self.source_path / transforms_file
        with open(transforms_path, encoding="utf-8") as handle:
            contents = json.load(handle)

        frames = contents["frames"]
        records: list[FrameRecord] = []
        for idx, frame in enumerate(frames):
            transform = np.array(frame["transform_matrix"])
            tmp_r = -transform[:3, :3]
            tmp_r[:, 0] = -tmp_r[:, 0]
            transform[:3, :3] = tmp_r
            matrix = np.linalg.inv(transform)

            r_matrix = np.transpose(matrix[:3, :3])
            t_vector = matrix[:3, 3]
            image_path = self.source_path / frame["file_path"]

            width = frame.get("w")
            height = frame.get("h")
            fl_x = frame.get("fl_x")
            fl_y = frame.get("fl_y")
            if width is None or height is None or fl_x is None or fl_y is None:
                image = Image.open(image_path)
                width, height = image.size
                fl_x = fl_x or width
                fl_y = fl_y or height

            cx = frame.get("cx", width / 2)
            cy = frame.get("cy", height / 2)
            params = CameraParams(
                uid=idx,
                width=int(width),
                height=int(height),
                fx=float(fl_x),
                fy=float(fl_y),
                cx=float(cx),
                cy=float(cy),
                R=r_matrix,
                T=t_vector,
            )
            records.append(
                FrameRecord(
                    frame_id=image_path.stem,
                    image_name=image_path.stem,
                    params=params,
                    assets={
                        "image": AssetRef(
                            kind="image",
                            path=image_path,
                            native_shape=(int(height), int(width)),
                            dtype="uint8",
                            codec=image_path.suffix,
                        )
                    },
                )
            )
        return records

    def read_scene_manifest(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        frames = self._read_transforms("transforms.json")
        frames.sort(key=lambda frame: frame.image_name)

        train_ids, test_ids = split_train_test(frames, eval=eval, llffhold=llffhold)
        scene_transform = compute_nerfpp_normalization([frames[i] for i in train_ids] if train_ids else frames)

        ply_path = self.source_path / "points3d.ply"
        if not ply_path.exists():
            num_pts = 100_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            storePly(str(ply_path), xyz, SH2RGB(shs) * 255)

        return SceneManifest(
            frames=tuple(frames),
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        return self.read_scene_manifest(eval=eval, llffhold=llffhold)
