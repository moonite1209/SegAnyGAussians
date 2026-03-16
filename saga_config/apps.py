from __future__ import annotations

"""Top-level app configuration bundles for each pipeline entrypoint."""

from .base import StrictConfigModel
from .clustering import ClusteringConfig
from .common import DatasetConfig, ModelConfig, PipeConfig
from .gui import GuiConfig
from .segment import SegmentConfig
from .training import TrainingConfig

__all__ = [
    "TrainingAppConfig",
    "ClusteringAppConfig",
    "GuiAppConfig",
    "SegmentAppConfig",
]


class SegmentAppConfig(StrictConfigModel):
    dataset: DatasetConfig
    segment: SegmentConfig


class TrainingAppConfig(StrictConfigModel):
    model: ModelConfig
    dataset: DatasetConfig
    pipe: PipeConfig
    training: TrainingConfig


class ClusteringAppConfig(StrictConfigModel):
    model: ModelConfig
    clustering: ClusteringConfig


class GuiAppConfig(StrictConfigModel):
    model: ModelConfig
    pipe: PipeConfig
    gui: GuiConfig
