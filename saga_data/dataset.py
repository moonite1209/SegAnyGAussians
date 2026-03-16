from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from torch.utils.data import Dataset

from .datastore import LocalDataStore
from .light_camera import LightCamera
from .resolution import compute_target_size
from .sample import FeatureFrameSample, RenderFrameSample
from .scene_index import SceneIndex
from .specs import CameraSpec, CameraParams


@dataclass(frozen=True)
class _FrameRecord:
    spec: CameraSpec
    target_size: tuple[int, int]
    scaled_params: CameraParams


class _BaseSceneDataset(Dataset):
    def __init__(
        self,
        scene_index: SceneIndex,
        indices: Optional[Sequence[int]] = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
    ):
        self.scene_index = scene_index
        selected_indices = list(indices) if indices is not None else list(range(len(scene_index.specs)))
        self._records: list[_FrameRecord] = []
        for idx in selected_indices:
            spec = scene_index.specs[idx]
            target_w, target_h = compute_target_size(
                spec.params.width,
                spec.params.height,
                resolution,
                resolution_scale,
            )
            self._records.append(
                _FrameRecord(
                    spec=spec,
                    target_size=(target_w, target_h),
                    scaled_params=spec.params.scaled_to(target_w, target_h),
                )
            )
        self.datastore = LocalDataStore()

    def __len__(self) -> int:
        return len(self._records)

    def _build_camera(self, params: CameraParams) -> LightCamera:
        return LightCamera.from_params(params, scene_transform=self.scene_index.scene_transform)

    def _load_render_sample(self, record: _FrameRecord) -> RenderFrameSample:
        image, alpha = self.datastore.load_image(record.spec.data_index.image_path, record.target_size)
        return RenderFrameSample(
            image=image,
            alpha=alpha,
            camera=self._build_camera(record.scaled_params),
            image_name=record.spec.image_name,
        )


class RenderDataset(_BaseSceneDataset):
    def __getitem__(self, idx: int) -> RenderFrameSample:
        return self._load_render_sample(self._records[idx])


class FeatureDataset(_BaseSceneDataset):
    def __init__(
        self,
        scene_index: SceneIndex,
        segment_masks_dir: str,
        segment_labels_dir: str,
        segment_label_features_path: str,
        indices: Optional[Sequence[int]] = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
    ):
        self.segment_masks_dir = Path(segment_masks_dir)
        self.segment_labels_dir = Path(segment_labels_dir)
        self.segment_label_features_path = Path(segment_label_features_path)
        super().__init__(
            scene_index=scene_index,
            indices=indices,
            resolution=resolution,
            resolution_scale=resolution_scale,
        )
        self._mask_paths: list[Path] = []
        self._label_paths: list[Path] = []
        self._validate_and_resolve_feature_paths()

    def _validate_and_resolve_feature_paths(self) -> None:
        if not self.segment_masks_dir.is_dir():
            raise FileNotFoundError(f"Segment masks directory not found: {self.segment_masks_dir}")
        if not self.segment_labels_dir.is_dir():
            raise FileNotFoundError(f"Segment labels directory not found: {self.segment_labels_dir}")
        if not self.segment_label_features_path.is_file():
            raise FileNotFoundError(f"Segment label features not found: {self.segment_label_features_path}")

        for record in self._records:
            mask_path = self.segment_masks_dir / f"{record.spec.image_name}.pt"
            label_path = self.segment_labels_dir / f"{record.spec.image_name}.pt"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask file not found for {record.spec.image_name}: {mask_path}")
            if not label_path.is_file():
                raise FileNotFoundError(f"Label file not found for {record.spec.image_name}: {label_path}")
            self._mask_paths.append(mask_path)
            self._label_paths.append(label_path)

    def __getitem__(self, idx: int) -> FeatureFrameSample:
        record = self._records[idx]
        render_sample = self._load_render_sample(record)
        masks = self.datastore.load_masks(self._mask_paths[idx], record.target_size)
        labels, label_features = self.datastore.load_labels(
            self._label_paths[idx],
            self.segment_label_features_path,
        )
        return FeatureFrameSample(
            image=render_sample.image,
            alpha=render_sample.alpha,
            camera=render_sample.camera,
            image_name=render_sample.image_name,
            masks=masks,
            labels=labels,
            label_features=label_features,
        )


def collate_list(batch):
    return batch
