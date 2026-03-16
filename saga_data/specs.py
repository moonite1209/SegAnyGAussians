from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import numpy as np


@dataclass(frozen=True)
class CameraParams:
    uid: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    T: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "R", self.R.view())
        self.R.setflags(write=False)
        object.__setattr__(self, "T", self.T.view())
        self.T.setflags(write=False)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Invalid focal length: fx={self.fx}, fy={self.fy}")

    @property
    def FoVx(self) -> float:
        return 2 * math.atan(self.width / (2 * self.fx))

    @property
    def FoVy(self) -> float:
        return 2 * math.atan(self.height / (2 * self.fy))

    def scaled_to(self, target_w: int, target_h: int) -> "CameraParams":
        sx = float(target_w) / float(self.width)
        sy = float(target_h) / float(self.height)
        return CameraParams(
            uid=self.uid,
            width=target_w,
            height=target_h,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            R=self.R,
            T=self.T,
        )


@dataclass(frozen=True)
class CameraDataIndex:
    image_path: Path
    mask_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    confidence_path: Optional[Path] = None

    def __post_init__(self):
        if not isinstance(self.image_path, Path):
            object.__setattr__(self, "image_path", Path(self.image_path))
        for field_name in ["mask_path", "labels_path", "depth_path", "confidence_path"]:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, field_name, Path(value))


@dataclass(frozen=True)
class CameraSpec:
    params: CameraParams
    data_index: CameraDataIndex
    image_name: str
