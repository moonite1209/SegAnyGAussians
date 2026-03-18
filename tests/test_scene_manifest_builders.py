from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from saga_data import build_scene_manifest


def _write_image(path: Path, size: tuple[int, int] = (8, 6), color: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    Image.fromarray(image, mode="RGB").save(path)


def _make_dataset_config(**kwargs):
    defaults = {
        "base_path": kwargs.get("base_path", ""),
        "images_path": kwargs.get("images_path", ""),
        "sparse_path": kwargs.get("sparse_path"),
        "source_path": kwargs.get("source_path"),
        "depth_path": kwargs.get("depth_path"),
        "eval": kwargs.get("eval", False),
        "llffhold": kwargs.get("llffhold", 8),
    }
    return SimpleNamespace(**defaults)


def test_build_scene_manifest_from_colmap_text(tmp_path):
    images_dir = tmp_path / "images"
    sparse_dir = tmp_path / "sparse"
    _write_image(images_dir / "frame_000.png")
    _write_image(images_dir / "frame_001.png", color=(32, 64, 128))
    sparse_dir.mkdir()
    (sparse_dir / "cameras.txt").write_text("1 PINHOLE 8 6 4.0 4.0 4.0 3.0\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text(
        (
            "1 1 0 0 0 0 0 0 1 frame_000.png\n0 0 -1\n"
            "2 1 0 0 0 1 0 0 1 frame_001.png\n0 0 -1\n"
        ),
        encoding="utf-8",
    )
    (sparse_dir / "points3D.txt").write_text("1 0 0 0 255 255 255 0.0\n", encoding="utf-8")

    manifest = build_scene_manifest(
        _make_dataset_config(
            base_path=str(tmp_path),
            images_path=str(images_dir),
            sparse_path=str(sparse_dir),
        )
    )

    assert len(manifest.frames) == 2
    assert manifest.frames[0].frame_id == "frame_000"
    assert manifest.frames[0].require_asset("image").path == images_dir / "frame_000.png"
    assert tuple(manifest.train_ids) == (0, 1)
    assert tuple(manifest.test_ids) == ()
    assert manifest.ply_path == sparse_dir / "points3D.ply"
    assert manifest.ply_path.is_file()


def test_build_scene_manifest_from_blender_train_test_layout(tmp_path):
    source_path = tmp_path
    _write_image(source_path / "train" / "r_001.png")
    _write_image(source_path / "train" / "r_002.png", color=(90, 80, 70))
    _write_image(source_path / "test" / "r_001.png", color=(10, 20, 30))
    transforms_train = {
        "camera_angle_x": 0.8,
        "frames": [
            {"transform_matrix": np.eye(4).tolist()},
            {"transform_matrix": np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).tolist()},
        ],
    }
    transforms_test = {
        "camera_angle_x": 0.8,
        "frames": [{"transform_matrix": np.eye(4).tolist()}],
    }
    (source_path / "transforms_train.json").write_text(json.dumps(transforms_train), encoding="utf-8")
    (source_path / "transforms_test.json").write_text(json.dumps(transforms_test), encoding="utf-8")

    manifest = build_scene_manifest(
        _make_dataset_config(
            base_path=str(tmp_path),
            images_path=str(source_path / "train"),
            source_path=str(source_path),
            eval=True,
        )
    )

    assert [frame.frame_id for frame in manifest.frames] == ["train__r_001", "train__r_002", "test__r_001"]
    assert [frame.image_name for frame in manifest.frames] == ["r_001", "r_002", "r_001"]
    assert tuple(manifest.train_ids) == (0, 1)
    assert tuple(manifest.test_ids) == (2,)
    assert manifest.ply_path == source_path / "points3d.ply"
    assert manifest.ply_path.is_file()


def test_build_scene_manifest_from_lerf_transforms_sorts_frames(tmp_path):
    source_path = tmp_path
    _write_image(source_path / "images" / "z.jpg")
    _write_image(source_path / "images" / "a.jpg")
    _write_image(source_path / "images" / "m.jpg")
    transforms = {
        "frames": [
            {
                "file_path": "images/z.jpg",
                "transform_matrix": np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).tolist(),
                "w": 8,
                "h": 6,
                "fl_x": 4.0,
                "fl_y": 4.0,
            },
            {
                "file_path": "images/a.jpg",
                "transform_matrix": np.eye(4).tolist(),
                "w": 8,
                "h": 6,
                "fl_x": 4.0,
                "fl_y": 4.0,
            },
            {
                "file_path": "images/m.jpg",
                "transform_matrix": np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).tolist(),
                "w": 8,
                "h": 6,
                "fl_x": 4.0,
                "fl_y": 4.0,
            },
        ]
    }
    (source_path / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")

    manifest = build_scene_manifest(
        _make_dataset_config(
            base_path=str(tmp_path),
            images_path=str(source_path / "images"),
            source_path=str(source_path),
            eval=True,
            llffhold=3,
        )
    )

    assert [frame.image_name for frame in manifest.frames] == ["a", "m", "z"]
    assert tuple(manifest.train_ids) == (1, 2)
    assert tuple(manifest.test_ids) == (0,)
    assert manifest.ply_path == source_path / "points3d.ply"
    assert manifest.ply_path.is_file()
