import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from scene.dataset_readers import storePly, fetchPly
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB

from ..specs import CameraParams, CameraDataIndex, CameraSpec
from ..scene_index import SceneIndex, split_train_test, compute_nerfpp_normalization


class LerfReader:
    def __init__(self, source_path: Path, extension: str = ".jpg"):
        self.source_path = Path(source_path)
        self.extension = extension

    def _read_transforms(self, transforms_file: str) -> List[CameraSpec]:
        transforms_path = self.source_path / transforms_file
        with open(transforms_path) as f:
            contents = json.load(f)

        frames = contents["frames"]
        specs = []
        for idx, frame in enumerate(frames):
            tmp = np.array(frame["transform_matrix"])
            tmp_R = tmp[:3, :3]
            tmp_R = -tmp_R
            tmp_R[:, 0] = -tmp_R[:, 0]
            tmp[:3, :3] = tmp_R
            matrix = np.linalg.inv(tmp)

            R = np.transpose(matrix[:3, :3])
            T = matrix[:3, 3]

            image_path = self.source_path / frame["file_path"]
            image_name = image_path.stem

            w = frame.get("w", None)
            h = frame.get("h", None)
            fl_x = frame.get("fl_x", None)
            fl_y = frame.get("fl_y", None)

            if w is None or h is None or fl_x is None or fl_y is None:
                image = Image.open(image_path)
                w, h = image.size
                fl_x = fl_x or w
                fl_y = fl_y or h

            cx = frame.get("cx", w / 2)
            cy = frame.get("cy", h / 2)

            params = CameraParams(
                uid=idx,
                width=int(w),
                height=int(h),
                fx=float(fl_x),
                fy=float(fl_y),
                cx=float(cx),
                cy=float(cy),
                R=R,
                T=T,
            )

            data_index = CameraDataIndex(image_path=image_path)
            specs.append(CameraSpec(params=params, data_index=data_index, image_name=image_name))

        return specs

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneIndex:
        specs = self._read_transforms("transforms.json")
        specs.sort(key=lambda s: s.image_name)

        train_ids, test_ids = split_train_test(specs, eval=eval, llffhold=llffhold)
        scene_transform = compute_nerfpp_normalization([specs[i] for i in train_ids] if train_ids else specs)

        ply_path = self.source_path / "points3d.ply"
        if not ply_path.exists():
            num_pts = 100_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(str(ply_path), xyz, SH2RGB(shs) * 255)
        else:
            pcd = fetchPly(str(ply_path))

        return SceneIndex(
            specs=specs,
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )
