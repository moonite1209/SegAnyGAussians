from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .light_camera import LightCamera

__all__ = [
    "RenderFrameSample",
    "FeatureFrameSample",
    "move_sample_to_device",
]


@dataclass
class RenderFrameSample:
    image: torch.Tensor
    alpha: Optional[torch.Tensor]
    camera: LightCamera
    image_name: str

    def to(self, device: str, non_blocking: bool = True) -> "RenderFrameSample":
        self.image = self.image.to(device, non_blocking=non_blocking)
        if self.alpha is not None:
            self.alpha = self.alpha.to(device, non_blocking=non_blocking)
        self.camera.to(device)
        return self


@dataclass
class FeatureFrameSample(RenderFrameSample):
    masks: torch.Tensor
    labels: torch.Tensor
    label_features: torch.Tensor

    def to(self, device: str, non_blocking: bool = True) -> "FeatureFrameSample":
        super().to(device, non_blocking=non_blocking)
        self.masks = self.masks.to(device, non_blocking=non_blocking)
        self.labels = self.labels.to(device, non_blocking=non_blocking)
        self.label_features = self.label_features.to(device, non_blocking=non_blocking)
        return self


SampleT = Union[RenderFrameSample, FeatureFrameSample]


def move_sample_to_device(sample: SampleT, device: str, non_blocking: bool = True) -> SampleT:
    return sample.to(device, non_blocking=non_blocking)
