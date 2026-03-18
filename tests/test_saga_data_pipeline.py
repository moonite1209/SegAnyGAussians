from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from saga_data import (
    ArtifactIndexer,
    AssetRef,
    FeatureArtifactLayout,
    FeatureDataset,
    FrameRecord,
    FrameTransformPipeline,
    MaskDataset,
    RenderDataset,
    SceneManifest,
    SceneTransform,
    build_feature_manifest,
    move_sample_to_device,
)
from saga_data.specs import CameraParams


def _write_rgba_image(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., 0] = 255
    image[..., 1] = 128
    image[..., 3] = 255
    Image.fromarray(image, mode="RGBA").save(path)


def _write_dmb(path: Path, array: np.ndarray, *, is_confidence: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
        payload = array.reshape(height, width, 1)
    else:
        height, width, channels = array.shape
        payload = array
    with open(path, "wb") as handle:
        handle.write(struct.pack("<iiii", 1, height, width, channels))
        if is_confidence:
            handle.write(payload.astype(np.uint8).tobytes())
        else:
            handle.write(payload.astype(np.float32).tobytes())


def _make_manifest(tmp_path: Path) -> SceneManifest:
    image_path = tmp_path / "images" / "frame_a.png"
    _write_rgba_image(image_path)
    params = CameraParams(
        uid=0,
        width=8,
        height=6,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=3.0,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )
    frame = FrameRecord(
        frame_id="frame_a",
        image_name="frame_a",
        params=params,
        assets={
            "image": AssetRef(
                kind="image",
                path=image_path,
                native_shape=(6, 8),
                dtype="uint8",
                codec=".png",
            )
        },
    )
    return SceneManifest(
        frames=(frame,),
        train_ids=(0,),
        test_ids=(),
        ply_path=None,
        scene_transform=SceneTransform(translate=np.zeros(3, dtype=np.float32), radius=1.0),
    )


def _write_feature_artifacts(root: Path, frame_ids: list[str]) -> None:
    masks_dir = root / "masks"
    labels_dir = root / "labels"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_features = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    torch.save(label_features, labels_dir / "label_features.pt")
    base_masks = torch.zeros((2, 6, 8), dtype=torch.bool)
    base_masks[0, :3, :4] = True
    base_masks[1, 3:, 4:] = True
    for frame_id in frame_ids:
        torch.save(base_masks, masks_dir / f"{frame_id}.pt")
        torch.save(torch.tensor([0, 1], dtype=torch.int64), labels_dir / f"{frame_id}.pt")


def _make_dataset_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        base_path=str(tmp_path),
        images_path=str(tmp_path / "train"),
        sparse_path=None,
        source_path=str(tmp_path),
        depth_path=None,
        eval=False,
        llffhold=8,
    )


def test_artifact_indexer_attaches_feature_assets_and_feature_dataset_reads_them(tmp_path):
    manifest = _make_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_feature_artifacts(artifacts_root, ["frame_a"])

    feature_manifest = ArtifactIndexer(
        FeatureArtifactLayout(root_dir=artifacts_root)
    ).attach_feature_artifacts(manifest)
    dataset = FeatureDataset(feature_manifest, split="train", resolution=2)
    sample = dataset[0]

    assert feature_manifest.frames[0].require_asset("masks").path == artifacts_root / "masks" / "frame_a.pt"
    assert feature_manifest.require_global_asset("label_features").path == artifacts_root / "labels" / "label_features.pt"
    assert sample.image.shape == (3, 3, 4)
    assert sample.alpha is not None and sample.alpha.shape == (1, 3, 4)
    assert sample.masks.shape == (2, 3, 4)
    assert sample.labels.tolist() == [0, 1]
    assert sample.label_features.shape == (3, 5)
    assert sample.camera.image_width == 4
    assert sample.camera.image_height == 3
    moved = move_sample_to_device(sample, "cpu")
    assert moved.image.device.type == "cpu"
    assert moved.masks.device.type == "cpu"


def test_mask_dataset_reads_manifest_attached_masks(tmp_path):
    manifest = _make_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_feature_artifacts(artifacts_root, ["frame_a"])
    mask_manifest = ArtifactIndexer(FeatureArtifactLayout(root_dir=artifacts_root)).attach_mask_artifacts(manifest)

    sample = MaskDataset(mask_manifest, split="train", resolution=1)[0]

    assert sample.image.shape == (3, 6, 8)
    assert sample.masks.shape == (2, 6, 8)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_render_dataset_supports_basic_dataloader_usage(tmp_path, num_workers):
    manifest = _make_manifest(tmp_path)
    dataset = RenderDataset(manifest, split="train", resolution=1)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)

    sample = next(iter(dataloader))
    assert sample.image_name == "frame_a"
    assert sample.image.shape == (3, 6, 8)


def test_artifact_indexer_rejects_orphan_mask_files(tmp_path):
    manifest = _make_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_feature_artifacts(artifacts_root, ["frame_a"])
    torch.save(torch.zeros((1, 6, 8), dtype=torch.bool), artifacts_root / "masks" / "ghost.pt")

    with pytest.raises(FileNotFoundError, match="orphan mask artifacts"):
        ArtifactIndexer(FeatureArtifactLayout(root_dir=artifacts_root)).attach_feature_artifacts(manifest)


def test_artifact_indexer_rejects_mask_label_count_mismatch(tmp_path):
    manifest = _make_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_feature_artifacts(artifacts_root, ["frame_a"])
    torch.save(torch.tensor([0], dtype=torch.int64), artifacts_root / "labels" / "frame_a.pt")

    with pytest.raises(ValueError, match="mismatched mask/label counts"):
        ArtifactIndexer(FeatureArtifactLayout(root_dir=artifacts_root)).attach_feature_artifacts(manifest)


def test_artifact_indexer_rejects_labels_out_of_range(tmp_path):
    manifest = _make_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_feature_artifacts(artifacts_root, ["frame_a"])
    torch.save(torch.tensor([0, 9], dtype=torch.int64), artifacts_root / "labels" / "frame_a.pt")

    with pytest.raises(ValueError, match="outside \\[0, 2\\]"):
        ArtifactIndexer(FeatureArtifactLayout(root_dir=artifacts_root)).attach_feature_artifacts(manifest)


def test_frame_transform_pipeline_resizes_depth_and_confidence_assets(tmp_path):
    manifest = _make_manifest(tmp_path)
    frame = manifest.frames[0]
    depth_path = tmp_path / "depth" / "frame_a_smoothDepth.dmb"
    confidence_path = tmp_path / "depth" / "frame_a_confidence.dmb"
    _write_dmb(depth_path, np.arange(48, dtype=np.float32).reshape(6, 8), is_confidence=False)
    _write_dmb(confidence_path, np.arange(48, dtype=np.uint8).reshape(6, 8), is_confidence=True)

    frame = frame.with_assets(
        {
            "depth": AssetRef(kind="depth", path=depth_path, codec=".dmb"),
            "confidence": AssetRef(kind="confidence", path=confidence_path, codec=".dmb"),
        }
    )
    manifest = manifest.with_frames((frame,))

    prepared = FrameTransformPipeline().prepare(
        manifest,
        manifest.frames[0],
        resolution=2,
        resolution_scale=1.0,
        required_assets=("image", "depth", "confidence"),
    )

    assert prepared.image.shape == (3, 3, 4)
    assert prepared.require_asset("depth").shape == (1, 3, 4)
    assert prepared.require_asset("confidence").shape == (1, 3, 4)
    assert prepared.scaled_params.width == 4
    assert prepared.scaled_params.height == 3


def test_disk_to_manifest_to_dataset_smoke_for_feature_training_pipeline(tmp_path):
    source_path = tmp_path
    _write_rgba_image(source_path / "train" / "r_001.png")
    _write_rgba_image(source_path / "train" / "r_002.png")
    transforms_train = {
        "camera_angle_x": 0.8,
        "frames": [
            {"transform_matrix": np.eye(4).tolist()},
            {"transform_matrix": np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).tolist()},
        ],
    }
    transforms_test = {"camera_angle_x": 0.8, "frames": []}
    (source_path / "transforms_train.json").write_text(json.dumps(transforms_train), encoding="utf-8")
    (source_path / "transforms_test.json").write_text(json.dumps(transforms_test), encoding="utf-8")

    artifacts_root = source_path / "saga"
    _write_feature_artifacts(artifacts_root, ["train__r_001", "train__r_002"])
    manifest = build_feature_manifest(_make_dataset_config(source_path), artifacts_dir=artifacts_root)
    dataset = FeatureDataset(manifest, split="train", resolution=1)
    sample = dataset[0]

    assert len(manifest.frames) == 2
    assert tuple(manifest.train_ids) == (0, 1)
    assert sample.image_name in {"r_001", "r_002"}
    assert sample.masks.shape == (2, 6, 8)
    assert sample.label_features.shape == (3, 5)
