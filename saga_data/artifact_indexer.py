from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .scene_index import AssetRef, FrameRecord, SceneManifest


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


@dataclass(frozen=True)
class FeatureArtifactLayout:
    root_dir: Path
    masks_dir_name: str = "masks"
    labels_dir_name: str = "labels"
    label_features_file_name: str = "label_features.pt"

    def __post_init__(self) -> None:
        if not isinstance(self.root_dir, Path):
            object.__setattr__(self, "root_dir", Path(self.root_dir))

    @property
    def masks_dir(self) -> Path:
        return self.root_dir / self.masks_dir_name

    @property
    def labels_dir(self) -> Path:
        return self.root_dir / self.labels_dir_name

    @property
    def label_features_path(self) -> Path:
        return self.labels_dir / self.label_features_file_name


class ArtifactIndexer:
    def __init__(self, layout: FeatureArtifactLayout):
        self.layout = layout

    def attach_mask_artifacts(self, manifest: SceneManifest) -> SceneManifest:
        if not self.layout.masks_dir.is_dir():
            raise FileNotFoundError(f"Feature masks directory not found: {self.layout.masks_dir}")

        mask_paths = self._collect_frame_files(self.layout.masks_dir)
        frame_ids = {frame.frame_id for frame in manifest.frames}
        orphan_masks = sorted(set(mask_paths) - frame_ids)
        if orphan_masks:
            raise FileNotFoundError(f"Found orphan mask artifacts with no matching frame ids: {orphan_masks}")

        updated_frames: list[FrameRecord] = []
        for frame in manifest.frames:
            mask_path = mask_paths.get(frame.frame_id)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing mask artifact for frame `{frame.frame_id}`: {self.layout.masks_dir / (frame.frame_id + '.pt')}"
                )
            masks = _torch_load(mask_path)
            if not isinstance(masks, torch.Tensor):
                raise TypeError(
                    f"Masks for frame `{frame.frame_id}` must be a torch.Tensor, got {type(masks).__name__}"
                )
            if masks.ndim != 3:
                raise ValueError(
                    f"Masks for frame `{frame.frame_id}` must have shape [N, H, W], got {tuple(masks.shape)}"
                )
            updated_frames.append(
                frame.with_asset(
                    "masks",
                    AssetRef(
                        kind="masks",
                        path=mask_path,
                        native_shape=tuple(int(dim) for dim in masks.shape),
                        dtype=str(masks.dtype),
                        codec=mask_path.suffix,
                    ),
                )
            )
        return manifest.with_frames(updated_frames)

    def attach_feature_artifacts(self, manifest: SceneManifest) -> SceneManifest:
        self._validate_layout()
        label_features = self._load_label_features()
        manifest_with_masks = self.attach_mask_artifacts(manifest)
        mask_paths = self._collect_frame_files(self.layout.masks_dir)
        label_paths = self._collect_frame_files(self.layout.labels_dir, exclude={self.layout.label_features_file_name})
        frame_ids = {frame.frame_id for frame in manifest_with_masks.frames}

        orphan_masks = sorted(set(mask_paths) - frame_ids)
        orphan_labels = sorted(set(label_paths) - frame_ids)
        if orphan_masks:
            raise FileNotFoundError(f"Found orphan mask artifacts with no matching frame ids: {orphan_masks}")
        if orphan_labels:
            raise FileNotFoundError(f"Found orphan label artifacts with no matching frame ids: {orphan_labels}")

        updated_frames: list[FrameRecord] = []
        for frame in manifest_with_masks.frames:
            mask_path = mask_paths.get(frame.frame_id)
            label_path = label_paths.get(frame.frame_id)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing mask artifact for frame `{frame.frame_id}`: {self.layout.masks_dir / (frame.frame_id + '.pt')}"
                )
            if label_path is None:
                raise FileNotFoundError(
                    f"Missing label artifact for frame `{frame.frame_id}`: {self.layout.labels_dir / (frame.frame_id + '.pt')}"
                )

            masks = _torch_load(mask_path)
            labels = _torch_load(label_path)
            self._validate_frame_artifacts(
                frame_id=frame.frame_id,
                masks=masks,
                labels=labels,
                num_label_features=label_features.shape[0],
            )
            updated_frames.append(
                frame.with_asset(
                    "labels",
                    AssetRef(
                        kind="labels",
                        path=label_path,
                        native_shape=tuple(int(dim) for dim in labels.shape),
                        dtype=str(labels.dtype),
                        codec=label_path.suffix,
                    ),
                )
            )

        global_assets = dict(manifest_with_masks.global_assets)
        global_assets["label_features"] = AssetRef(
            kind="label_features",
            path=self.layout.label_features_path,
            native_shape=tuple(int(dim) for dim in label_features.shape),
            dtype=str(label_features.dtype),
            codec=self.layout.label_features_path.suffix,
        )
        return manifest_with_masks.with_frames(updated_frames).with_global_assets(global_assets)

    def _validate_layout(self) -> None:
        if not self.layout.masks_dir.is_dir():
            raise FileNotFoundError(f"Feature masks directory not found: {self.layout.masks_dir}")
        if not self.layout.labels_dir.is_dir():
            raise FileNotFoundError(f"Feature labels directory not found: {self.layout.labels_dir}")
        if not self.layout.label_features_path.is_file():
            raise FileNotFoundError(f"Label features file not found: {self.layout.label_features_path}")

    def _load_label_features(self) -> torch.Tensor:
        label_features = _torch_load(self.layout.label_features_path)
        if not isinstance(label_features, torch.Tensor):
            raise TypeError(
                f"Expected label features to be a torch.Tensor, got {type(label_features).__name__}"
            )
        if label_features.ndim != 2:
            raise ValueError(
                f"Label features must have shape [num_labels, feature_dim], got {tuple(label_features.shape)}"
            )
        return label_features

    @staticmethod
    def _collect_frame_files(directory: Path, exclude: set[str] | None = None) -> dict[str, Path]:
        exclude = exclude or set()
        frame_files: dict[str, Path] = {}
        for path in sorted(directory.glob("*.pt")):
            if path.name in exclude:
                continue
            frame_id = path.stem
            if frame_id in frame_files:
                raise ValueError(f"Duplicate artifact file for frame `{frame_id}` in {directory}")
            frame_files[frame_id] = path
        return frame_files

    @staticmethod
    def _validate_frame_artifacts(
        *,
        frame_id: str,
        masks,
        labels,
        num_label_features: int,
    ) -> None:
        if not isinstance(masks, torch.Tensor):
            raise TypeError(f"Masks for frame `{frame_id}` must be a torch.Tensor, got {type(masks).__name__}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Labels for frame `{frame_id}` must be a torch.Tensor, got {type(labels).__name__}")
        if masks.ndim != 3:
            raise ValueError(f"Masks for frame `{frame_id}` must have shape [N, H, W], got {tuple(masks.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"Labels for frame `{frame_id}` must have shape [N], got {tuple(labels.shape)}")
        if masks.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Frame `{frame_id}` has mismatched mask/label counts: {masks.shape[0]} masks vs {labels.shape[0]} labels"
            )
        if labels.numel() == 0:
            return
        min_label = int(labels.min().item())
        max_label = int(labels.max().item())
        if min_label < 0 or max_label >= num_label_features:
            raise ValueError(
                f"Frame `{frame_id}` has label ids outside [0, {num_label_features - 1}]: "
                f"min={min_label}, max={max_label}"
            )
