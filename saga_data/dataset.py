from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from torch.utils.data import Dataset

from .datastore import LocalDataStoreV2
from .pipeline import FrameTransformPipeline, PreparedFrame
from .sample import FeatureFrameSample, MaskFrameSample, RenderFrameSample
from .scene_index import FrameRecord, SceneManifest


SampleBuilder = Callable[[PreparedFrame], object]


def _identity_sample(prepared: PreparedFrame):
    return prepared


@dataclass(frozen=True)
class _DatasetFrame:
    index: int
    record: FrameRecord


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: SceneManifest,
        *,
        indices: Optional[Sequence[int]] = None,
        split: str | None = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
        required_assets: Iterable[str] = ("image",),
        required_global_assets: Iterable[str] = (),
        sample_builder: SampleBuilder | None = None,
        datastore: LocalDataStoreV2 | None = None,
        transform_pipeline: FrameTransformPipeline | None = None,
    ):
        if indices is not None and split is not None:
            raise ValueError("Pass either `indices` or `split`, not both")
        self.manifest = manifest
        self.resolution = resolution
        self.resolution_scale = resolution_scale
        self.required_assets = tuple(required_assets)
        self.required_global_assets = tuple(required_global_assets)
        self.sample_builder = sample_builder or _identity_sample
        self.datastore = datastore or LocalDataStoreV2()
        self.transform_pipeline = transform_pipeline or FrameTransformPipeline(self.datastore)

        if split is not None:
            if split == "train":
                selected_indices = list(manifest.train_ids)
            elif split == "test":
                selected_indices = list(manifest.test_ids)
            else:
                raise ValueError(f"Unsupported split `{split}`")
        else:
            selected_indices = list(indices) if indices is not None else list(range(len(manifest.frames)))

        self._frames = [
            _DatasetFrame(index=index, record=manifest.frames[index])
            for index in selected_indices
        ]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        frame = self._frames[idx]
        prepared = self.transform_pipeline.prepare(
            self.manifest,
            frame.record,
            resolution=self.resolution,
            resolution_scale=self.resolution_scale,
            required_assets=self.required_assets,
            required_global_assets=self.required_global_assets,
        )
        return self.sample_builder(prepared)


def build_render_sample(prepared: PreparedFrame) -> RenderFrameSample:
    return RenderFrameSample(
        image=prepared.image,
        alpha=prepared.alpha,
        camera=prepared.camera,
        image_name=prepared.record.image_name,
    )


def build_mask_sample(prepared: PreparedFrame) -> MaskFrameSample:
    return MaskFrameSample(
        image=prepared.image,
        alpha=prepared.alpha,
        camera=prepared.camera,
        image_name=prepared.record.image_name,
        masks=prepared.require_asset("masks"),
    )


def build_feature_sample(prepared: PreparedFrame) -> FeatureFrameSample:
    return FeatureFrameSample(
        image=prepared.image,
        alpha=prepared.alpha,
        camera=prepared.camera,
        image_name=prepared.record.image_name,
        masks=prepared.require_asset("masks"),
        labels=prepared.require_asset("labels"),
        label_features=prepared.require_global_asset("label_features"),
    )


class RenderDataset(ManifestDataset):
    def __init__(
        self,
        manifest: SceneManifest,
        indices: Optional[Sequence[int]] = None,
        *,
        split: str | None = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
        datastore: LocalDataStoreV2 | None = None,
        transform_pipeline: FrameTransformPipeline | None = None,
    ):
        super().__init__(
            manifest,
            indices=indices,
            split=split,
            resolution=resolution,
            resolution_scale=resolution_scale,
            required_assets=("image",),
            sample_builder=build_render_sample,
            datastore=datastore,
            transform_pipeline=transform_pipeline,
        )


class FeatureDataset(ManifestDataset):
    def __init__(
        self,
        manifest: SceneManifest,
        indices: Optional[Sequence[int]] = None,
        *,
        split: str | None = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
        datastore: LocalDataStoreV2 | None = None,
        transform_pipeline: FrameTransformPipeline | None = None,
    ):
        super().__init__(
            manifest,
            indices=indices,
            split=split,
            resolution=resolution,
            resolution_scale=resolution_scale,
            required_assets=("image", "masks", "labels"),
            required_global_assets=("label_features",),
            sample_builder=build_feature_sample,
            datastore=datastore,
            transform_pipeline=transform_pipeline,
        )


class MaskDataset(ManifestDataset):
    def __init__(
        self,
        manifest: SceneManifest,
        indices: Optional[Sequence[int]] = None,
        *,
        split: str | None = None,
        resolution: int = 1,
        resolution_scale: float = 1.0,
        datastore: LocalDataStoreV2 | None = None,
        transform_pipeline: FrameTransformPipeline | None = None,
    ):
        super().__init__(
            manifest,
            indices=indices,
            split=split,
            resolution=resolution,
            resolution_scale=resolution_scale,
            required_assets=("image", "masks"),
            sample_builder=build_mask_sample,
            datastore=datastore,
            transform_pipeline=transform_pipeline,
        )


def collate_list(batch):
    return batch
