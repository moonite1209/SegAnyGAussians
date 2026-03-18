from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .specs import CameraParams


AssetDict = Mapping[str, "AssetRef"]


@dataclass(frozen=True)
class AssetRef:
    kind: str
    path: Path
    native_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    codec: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if self.native_shape is not None:
            object.__setattr__(self, "native_shape", tuple(int(dim) for dim in self.native_shape))


@dataclass(frozen=True)
class SceneTransform:
    translate: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "translate", np.asarray(self.translate, dtype=np.float32).view())
        self.translate.setflags(write=False)
        if self.radius <= 0:
            raise ValueError(f"Invalid scene radius: {self.radius}")


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    image_name: str
    params: CameraParams
    assets: AssetDict

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("`frame_id` must be non-empty")
        if not self.image_name:
            raise ValueError("`image_name` must be non-empty")
        object.__setattr__(self, "assets", dict(self.assets))

    def require_asset(self, name: str) -> AssetRef:
        asset = self.assets.get(name)
        if asset is None:
            raise KeyError(f"Frame `{self.frame_id}` is missing required asset `{name}`")
        return asset

    def with_asset(self, name: str, asset: AssetRef) -> "FrameRecord":
        new_assets = dict(self.assets)
        new_assets[name] = asset
        return FrameRecord(
            frame_id=self.frame_id,
            image_name=self.image_name,
            params=self.params,
            assets=new_assets,
        )

    def with_assets(self, assets: Mapping[str, AssetRef]) -> "FrameRecord":
        new_assets = dict(self.assets)
        new_assets.update(assets)
        return FrameRecord(
            frame_id=self.frame_id,
            image_name=self.image_name,
            params=self.params,
            assets=new_assets,
        )


@dataclass(frozen=True)
class SceneManifest:
    frames: tuple[FrameRecord, ...]
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    ply_path: Optional[Path]
    scene_transform: SceneTransform
    global_assets: AssetDict = field(default_factory=dict)
    label_vocabulary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "train_ids", tuple(int(idx) for idx in self.train_ids))
        object.__setattr__(self, "test_ids", tuple(int(idx) for idx in self.test_ids))
        if self.ply_path is not None and not isinstance(self.ply_path, Path):
            object.__setattr__(self, "ply_path", Path(self.ply_path))
        object.__setattr__(self, "global_assets", dict(self.global_assets))
        object.__setattr__(self, "label_vocabulary", tuple(self.label_vocabulary))

        seen: set[str] = set()
        duplicates: set[str] = set()
        for frame in self.frames:
            if frame.frame_id in seen:
                duplicates.add(frame.frame_id)
            seen.add(frame.frame_id)
        if duplicates:
            raise ValueError(f"Duplicate frame ids found in manifest: {sorted(duplicates)}")

        frame_count = len(self.frames)
        for split_name, split_ids in (("train", self.train_ids), ("test", self.test_ids)):
            invalid = [idx for idx in split_ids if idx < 0 or idx >= frame_count]
            if invalid:
                raise ValueError(f"Invalid {split_name} split indices: {invalid}")

    @property
    def specs(self) -> tuple[FrameRecord, ...]:
        return self.frames

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, index: int) -> FrameRecord:
        return self.frames[index]

    def frame_by_id(self, frame_id: str) -> FrameRecord:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(f"Unknown frame id: {frame_id}")

    def require_global_asset(self, name: str) -> AssetRef:
        asset = self.global_assets.get(name)
        if asset is None:
            raise KeyError(f"Manifest is missing required global asset `{name}`")
        return asset

    def with_frames(self, frames: Sequence[FrameRecord]) -> "SceneManifest":
        return SceneManifest(
            frames=tuple(frames),
            train_ids=self.train_ids,
            test_ids=self.test_ids,
            ply_path=self.ply_path,
            scene_transform=self.scene_transform,
            global_assets=self.global_assets,
            label_vocabulary=self.label_vocabulary,
        )

    def with_global_assets(self, assets: Mapping[str, AssetRef]) -> "SceneManifest":
        return SceneManifest(
            frames=self.frames,
            train_ids=self.train_ids,
            test_ids=self.test_ids,
            ply_path=self.ply_path,
            scene_transform=self.scene_transform,
            global_assets=assets,
            label_vocabulary=self.label_vocabulary,
        )

    def train_specs(self) -> list[FrameRecord]:
        return [self.frames[i] for i in self.train_ids]

    def test_specs(self) -> list[FrameRecord]:
        return [self.frames[i] for i in self.test_ids]


SceneIndex = SceneManifest


def split_train_test(
    frames: Sequence[FrameRecord],
    eval: bool = False,
    llffhold: int = 8,
) -> tuple[list[int], list[int]]:
    if not eval:
        return list(range(len(frames))), []
    train_ids = [i for i in range(len(frames)) if i % llffhold != 0]
    test_ids = [i for i in range(len(frames)) if i % llffhold == 0]
    return train_ids, test_ids


def compute_nerfpp_normalization(frames: Sequence[FrameRecord]) -> SceneTransform:
    from utils.graphics_utils import getWorld2View2

    cam_centers = []
    for frame in frames:
        params = frame.params
        w2c = getWorld2View2(params.R, params.T)
        c2w = np.linalg.inv(w2c)
        cam_centers.append(c2w[:3, 3:4])

    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    translate = -center.flatten()
    return SceneTransform(translate=translate, radius=float(radius))
