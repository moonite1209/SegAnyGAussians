from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .specs import CameraSpec, CameraParams


@dataclass(frozen=True)
class SceneTransform:
    translate: np.ndarray
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "translate", np.asarray(self.translate, dtype=np.float32).view())
        self.translate.setflags(write=False)
        if self.radius <= 0:
            raise ValueError(f"Invalid scene radius: {self.radius}")


@dataclass
class SceneIndex:
    specs: List[CameraSpec]
    train_ids: List[int]
    test_ids: List[int]
    ply_path: Optional[Path]
    scene_transform: SceneTransform

    def train_specs(self) -> List[CameraSpec]:
        return [self.specs[i] for i in self.train_ids]

    def test_specs(self) -> List[CameraSpec]:
        return [self.specs[i] for i in self.test_ids]


def split_train_test(
    specs: List[CameraSpec],
    eval: bool = False,
    llffhold: int = 8
) -> Tuple[List[int], List[int]]:
    if not eval:
        return list(range(len(specs))), []
    train_ids = [i for i in range(len(specs)) if i % llffhold != 0]
    test_ids = [i for i in range(len(specs)) if i % llffhold == 0]
    return train_ids, test_ids


def compute_nerfpp_normalization(specs: List[CameraSpec]) -> SceneTransform:
    from utils.graphics_utils import getWorld2View2

    cam_centers = []
    for spec in specs:
        params: CameraParams = spec.params
        W2C = getWorld2View2(params.R, params.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    translate = -center.flatten()
    return SceneTransform(translate=translate, radius=float(radius))
