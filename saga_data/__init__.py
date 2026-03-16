from __future__ import annotations

from pathlib import Path
from typing import Any

from saga_config import DatasetConfig

from .dataset import FeatureDataset, RenderDataset
from .geometry import depth_to_camera_points
from .readers.blender import BlenderReader
from .readers.colmap import ColmapReader
from .readers.lerf import LerfReader
from .sample import FeatureFrameSample, RenderFrameSample, move_sample_to_device
from .scene_index import SceneIndex, SceneTransform

__all__ = [
    "build_scene_index",
    "FeatureDataset",
    "RenderDataset",
    "FeatureFrameSample",
    "RenderFrameSample",
    "move_sample_to_device",
    "SceneIndex",
    "SceneTransform",
    "depth_to_camera_points",
]


def build_scene_index(dataset: DatasetConfig | Any) -> SceneIndex:
    """Build scene metadata from a validated dataset config."""

    sparse_path_value = getattr(dataset, "sparse_path", None)
    source_path_value = getattr(dataset, "source_path", None)
    images_path_value = getattr(dataset, "images_path", getattr(dataset, "images", None))
    depth_path_value = getattr(dataset, "depth_path", None)
    eval_value = getattr(dataset, "eval", False)
    llffhold_value = getattr(dataset, "llffhold", 8)

    sparse_path = Path(sparse_path_value) if sparse_path_value else None
    source_path = Path(source_path_value) if source_path_value else None

    if sparse_path and sparse_path.exists():
        reader = ColmapReader(
            sparse_path=str(sparse_path),
            images_path=images_path_value,
            depth_path=depth_path_value,
        )
        return reader.read_scene_index(eval=eval_value, llffhold=llffhold_value)

    if source_path and (source_path / "transforms_train.json").exists():
        reader = BlenderReader(source_path=str(source_path))
        return reader.read_scene_index(eval=eval_value, llffhold=llffhold_value)

    if source_path and (source_path / "transforms.json").exists():
        reader = LerfReader(source_path=str(source_path))
        return reader.read_scene_index(eval=eval_value, llffhold=llffhold_value)

    raise ValueError(
        "Could not recognize dataset type from dataset config: "
        f"sparse_path={sparse_path_value!r}, source_path={source_path_value!r}"
    )
