from __future__ import annotations

from pathlib import Path
from typing import Any

from saga_config import DatasetConfig

from .artifact_indexer import ArtifactIndexer, FeatureArtifactLayout
from .readers.blender import BlenderReader
from .readers.colmap import ColmapReader
from .readers.lerf import LerfReader
from .scene_index import SceneManifest


def build_scene_manifest(dataset: DatasetConfig | Any) -> SceneManifest:
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
            sparse_path=sparse_path,
            images_path=Path(images_path_value),
            depth_path=Path(depth_path_value) if depth_path_value else None,
        )
        return reader.read_scene_manifest(eval=eval_value, llffhold=llffhold_value)

    if source_path and (source_path / "transforms_train.json").exists():
        reader = BlenderReader(source_path=source_path)
        return reader.read_scene_manifest(eval=eval_value, llffhold=llffhold_value)

    if source_path and (source_path / "transforms.json").exists():
        reader = LerfReader(source_path=source_path)
        return reader.read_scene_manifest(eval=eval_value, llffhold=llffhold_value)

    raise ValueError(
        "Could not recognize dataset type from dataset config: "
        f"sparse_path={sparse_path_value!r}, source_path={source_path_value!r}"
    )


def infer_feature_artifacts_dir(
    *,
    artifacts_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    labels_dir: str | Path | None = None,
    label_features_path: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if artifacts_dir:
        candidates.append(Path(artifacts_dir))
    if masks_dir:
        mask_parent = Path(masks_dir).parent
        candidates.append(mask_parent)
    if labels_dir:
        label_parent = Path(labels_dir).parent
        candidates.append(label_parent)
    if label_features_path:
        label_features_parent = Path(label_features_path).parent.parent
        candidates.append(label_features_parent)

    if not candidates:
        raise ValueError(
            "Could not infer feature artifacts directory. Provide `artifacts_dir` or a compatible masks/labels layout."
        )

    normalized = [candidate.resolve() for candidate in candidates]
    first = normalized[0]
    mismatched = [str(candidate) for candidate in normalized[1:] if candidate != first]
    if mismatched:
        raise ValueError(
            "Feature artifact paths do not agree on a common root directory: "
            f"{[str(path) for path in normalized]}"
        )
    return first


def build_feature_manifest(
    dataset: DatasetConfig | Any,
    *,
    artifacts_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    labels_dir: str | Path | None = None,
    label_features_path: str | Path | None = None,
) -> SceneManifest:
    manifest = build_scene_manifest(dataset)
    root_dir = infer_feature_artifacts_dir(
        artifacts_dir=artifacts_dir,
        masks_dir=masks_dir,
        labels_dir=labels_dir,
        label_features_path=label_features_path,
    )
    return ArtifactIndexer(FeatureArtifactLayout(root_dir=root_dir)).attach_feature_artifacts(manifest)


def build_mask_manifest(
    dataset: DatasetConfig | Any,
    *,
    artifacts_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
) -> SceneManifest:
    manifest = build_scene_manifest(dataset)
    root_dir = infer_feature_artifacts_dir(
        artifacts_dir=artifacts_dir,
        masks_dir=masks_dir,
    )
    return ArtifactIndexer(FeatureArtifactLayout(root_dir=root_dir)).attach_mask_artifacts(manifest)


build_scene_index = build_scene_manifest
