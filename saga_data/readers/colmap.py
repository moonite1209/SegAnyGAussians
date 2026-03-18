from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.dataset_readers import storePly

from ..scene_index import AssetRef, FrameRecord, SceneManifest, compute_nerfpp_normalization, split_train_test
from ..specs import CameraParams


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
        if txt_path.exists():
            return read_extrinsics_text(str(txt_path))
        raise FileNotFoundError(f"COLMAP images file not found: {bin_path} or {txt_path}")

    def _read_intrinsics(self):
        bin_path = self.sparse_path / "cameras.bin"
        txt_path = self.sparse_path / "cameras.txt"
        if bin_path.exists():
            return read_intrinsics_binary(str(bin_path))
        if txt_path.exists():
            return read_intrinsics_text(str(txt_path))
        raise FileNotFoundError(f"COLMAP cameras file not found: {bin_path} or {txt_path}")

    def _create_frame_record(self, extr, intr) -> FrameRecord:
        height = int(intr.height)
        width = int(intr.width)
        r_matrix = np.transpose(qvec2rotmat(extr.qvec))
        t_vector = np.array(extr.tvec)

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

        image_name = Path(extr.name).stem
        image_path = self.images_path / extr.name
        params = CameraParams(
            uid=int(intr.id),
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            R=r_matrix,
            T=t_vector,
        )

        assets: dict[str, AssetRef] = {
            "image": AssetRef(
                kind="image",
                path=image_path,
                native_shape=(height, width),
                dtype="uint8",
                codec=image_path.suffix,
            )
        }
        if self.depth_path is not None:
            image_idx = image_name.split("-")[-1]
            depth_path = self.depth_path / f"{image_idx}_smoothDepth.dmb"
            confidence_path = self.depth_path / f"{image_idx}_confidence.dmb"
            assets["depth"] = AssetRef(kind="depth", path=depth_path, dtype="float32", codec=depth_path.suffix)
            assets["confidence"] = AssetRef(
                kind="confidence",
                path=confidence_path,
                dtype="uint8",
                codec=confidence_path.suffix,
            )

        return FrameRecord(
            frame_id=image_name,
            image_name=image_name,
            params=params,
            assets=assets,
        )

    def read_scene_manifest(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        frames: list[FrameRecord] = []
        for key in self.cam_extrinsics:
            extr = self.cam_extrinsics[key]
            intr = self.cam_intrinsics[extr.camera_id]
            frames.append(self._create_frame_record(extr, intr))

        frames.sort(key=lambda frame: frame.image_name)
        train_ids, test_ids = split_train_test(frames, eval=eval, llffhold=llffhold)
        scene_transform = compute_nerfpp_normalization([frames[i] for i in train_ids] if train_ids else frames)

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

        return SceneManifest(
            frames=tuple(frames),
            train_ids=train_ids,
            test_ids=test_ids,
            ply_path=ply_path,
            scene_transform=scene_transform,
        )

    def read_scene_index(self, eval: bool = False, llffhold: int = 8) -> SceneManifest:
        return self.read_scene_manifest(eval=eval, llffhold=llffhold)
