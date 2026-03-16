from pathlib import Path
from typing import Optional, List
import numpy as np

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
    qvec2rotmat,
)
from scene.dataset_readers import fetchPly, storePly

from ..specs import CameraParams, CameraDataIndex, CameraSpec
from ..scene_index import SceneIndex, split_train_test, compute_nerfpp_normalization


class ColmapReader:
    def __init__(
        self,
        sparse_path: Path,
        images_path: Path,
        depth_path: Optional[Path] = None,
    ):
        self.sparse_path = Path(sparse_path)
        self.images_path = Path(images_path)
        self.depth_path = Path(depth_path) if depth_path else None

        self.cam_extrinsics = self._read_extrinsics()
        self.cam_intrinsics = self._read_intrinsics()

    def _read_extrinsics(self):
        bin_path = self.sparse_path / "images.bin"
        txt_path = self.sparse_path / "images.txt"
        if bin_path.exists():
            return read_extrinsics_binary(str(bin_path))
        elif txt_path.exists():
            return read_extrinsics_text(str(txt_path))
        else:
            raise FileNotFoundError(f"COLMAP images file not found: {bin_path} or {txt_path}")

    def _read_intrinsics(self):
        bin_path = self.sparse_path / "cameras.bin"
        txt_path = self.sparse_path / "cameras.txt"
        if bin_path.exists():
            return read_intrinsics_binary(str(bin_path))
        elif txt_path.exists():
            return read_intrinsics_text(str(txt_path))
        else:
            raise FileNotFoundError(f"COLMAP cameras file not found: {bin_path} or {txt_path}")

    def _create_camera_spec(self, extr, intr, idx: int) -> CameraSpec:
        uid = intr.id
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            fx = intr.params[0]
            fy = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        elif intr.model == "SIMPLE_RADIAL":
            fx = intr.params[0]
            fy = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

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

        image_name_noext = Path(extr.name).stem
        image_path = self.images_path / extr.name

        if self.depth_path:
            image_idx = image_name_noext.split("-")[-1]
            depth_path = self.depth_path / f"{image_idx}_smoothDepth.dmb"
            confidence_path = self.depth_path / f"{image_idx}_confidence.dmb"
        else:
            depth_path = None
            confidence_path = None

        data_index = CameraDataIndex(
            image_path=image_path,
            depth_path=depth_path,
            confidence_path=confidence_path,
        )
        return CameraSpec(params=params, data_index=data_index, image_name=image_name_noext)

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneIndex:
        specs: List[CameraSpec] = []
        for key in self.cam_extrinsics:
            extr = self.cam_extrinsics[key]
            intr = self.cam_intrinsics[extr.camera_id]
            specs.append(self._create_camera_spec(extr, intr, len(specs)))

        specs.sort(key=lambda s: s.image_name)
        train_ids, test_ids = split_train_test(specs, eval=eval, llffhold=llffhold)
        scene_transform = compute_nerfpp_normalization([specs[i] for i in train_ids] if train_ids else specs)

        ply_path = self.sparse_path / "points3D.ply"
        bin_path = self.sparse_path / "points3D.bin"
        txt_path = self.sparse_path / "points3D.txt"
        if not ply_path.exists():
            if bin_path.exists():
                xyz, rgb, _ = read_points3D_binary(str(bin_path))
            elif txt_path.exists():
                xyz, rgb, _ = read_points3D_text(str(txt_path))
            else:
                raise FileNotFoundError(f"COLMAP points3D file not found: {bin_path} or {txt_path}")
            storePly(str(ply_path), xyz, rgb)

        return SceneIndex(
            specs=specs,
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )
