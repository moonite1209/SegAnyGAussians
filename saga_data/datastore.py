from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from utils.general_utils import PILtoTorch

from .dmb import read_dmb_file
from .scene_index import AssetRef
from .transforms import resize_alpha_tensor, resize_depth_tensor, resize_mask_tensor


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


@dataclass(frozen=True)
class DecodedImage:
    rgb: torch.Tensor
    alpha: torch.Tensor | None


LoaderFn = Callable[[AssetRef], object]


class AssetLoaderRegistry:
    def __init__(self):
        self._loaders: dict[str, LoaderFn] = {}

    def register(self, kind: str, loader: LoaderFn) -> None:
        self._loaders[kind] = loader

    def load(self, asset: AssetRef) -> object:
        loader = self._loaders.get(asset.kind)
        if loader is None:
            raise KeyError(f"No asset loader registered for kind `{asset.kind}`")
        return loader(asset)


def _decode_image(asset: AssetRef) -> DecodedImage:
    image = Image.open(asset.path)
    tensor = PILtoTorch(image, image.size)
    rgb = tensor[:3, ...].float()
    alpha = tensor[3:4, ...].float() if tensor.shape[0] == 4 else None
    return DecodedImage(rgb=rgb, alpha=alpha)


def _decode_tensor(asset: AssetRef) -> torch.Tensor:
    tensor = _torch_load(asset.path)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected `{asset.path}` to contain a torch.Tensor, got {type(tensor).__name__}")
    return tensor


def _decode_depth(asset: AssetRef) -> torch.Tensor:
    depth_array = read_dmb_file(asset.path, is_confidence=False)
    return torch.from_numpy(depth_array.copy())[None, ...]


def _decode_confidence(asset: AssetRef) -> torch.Tensor:
    confidence_array = read_dmb_file(asset.path, is_confidence=True)
    return torch.from_numpy(confidence_array.copy())[None, ...]


def create_default_registry() -> AssetLoaderRegistry:
    registry = AssetLoaderRegistry()
    registry.register("image", _decode_image)
    registry.register("masks", _decode_tensor)
    registry.register("labels", _decode_tensor)
    registry.register("label_features", _decode_tensor)
    registry.register("depth", _decode_depth)
    registry.register("confidence", _decode_confidence)
    return registry


class LocalDataStoreV2:
    """
    Local filesystem datastore with per-worker LRU cache for decoded assets.
    """

    def __init__(
        self,
        max_cache_bytes: int = 512 * 1024 * 1024,
        registry: AssetLoaderRegistry | None = None,
    ):
        self._max_cache_bytes = max_cache_bytes
        self._cache: "OrderedDict[str, tuple[object, int]]" = OrderedDict()
        self._cache_bytes = 0
        self.registry = registry or create_default_registry()

    def _cache_key(self, asset: AssetRef) -> str:
        return f"{asset.kind}::{asset.path}"

    def _estimate_size(self, value: object) -> int:
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        if isinstance(value, DecodedImage):
            size = value.rgb.element_size() * value.rgb.nelement()
            if value.alpha is not None:
                size += value.alpha.element_size() * value.alpha.nelement()
            return size
        return 0

    def _cache_put(self, key: str, value: object) -> None:
        if self._max_cache_bytes <= 0:
            return
        size = self._estimate_size(value)
        if size <= 0 or size > self._max_cache_bytes:
            return
        while self._cache_bytes + size > self._max_cache_bytes and self._cache:
            _, (_, evict_size) = self._cache.popitem(last=False)
            self._cache_bytes -= evict_size
        self._cache[key] = (value, size)
        self._cache_bytes += size

    def _cache_get(self, key: str) -> object | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        value, size = cached
        self._cache.move_to_end(key)
        return value

    def load_asset(self, asset: AssetRef) -> object:
        key = self._cache_key(asset)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        value = self.registry.load(asset)
        self._cache_put(key, value)
        return value

    def load_image(self, path_or_asset: Path | AssetRef) -> DecodedImage:
        asset = path_or_asset if isinstance(path_or_asset, AssetRef) else AssetRef(kind="image", path=Path(path_or_asset))
        decoded = self.load_asset(asset)
        if not isinstance(decoded, DecodedImage):
            raise TypeError(f"Decoded image loader returned {type(decoded).__name__}")
        return decoded

    def load_masks(self, path_or_asset: Path | AssetRef, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        asset = path_or_asset if isinstance(path_or_asset, AssetRef) else AssetRef(kind="masks", path=Path(path_or_asset))
        masks = self.load_asset(asset)
        if not isinstance(masks, torch.Tensor):
            raise TypeError(f"Decoded masks loader returned {type(masks).__name__}")
        masks = masks.bool()
        if target_size is None:
            return masks
        return resize_mask_tensor(masks, target_size)

    def load_label_features(self, path_or_asset: Path | AssetRef) -> torch.Tensor:
        asset = (
            path_or_asset
            if isinstance(path_or_asset, AssetRef)
            else AssetRef(kind="label_features", path=Path(path_or_asset))
        )
        label_features = self.load_asset(asset)
        if not isinstance(label_features, torch.Tensor):
            raise TypeError(f"Decoded label feature loader returned {type(label_features).__name__}")
        return label_features

    def load_labels(
        self,
        path_or_asset: Path | AssetRef,
        label_features_path_or_asset: Path | AssetRef | None = None,
    ):
        asset = path_or_asset if isinstance(path_or_asset, AssetRef) else AssetRef(kind="labels", path=Path(path_or_asset))
        labels = self.load_asset(asset)
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Decoded labels loader returned {type(labels).__name__}")
        if label_features_path_or_asset is None:
            return labels
        label_features = self.load_label_features(label_features_path_or_asset)
        return labels, label_features

    def load_depth(self, path_or_asset: Path | AssetRef, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        asset = path_or_asset if isinstance(path_or_asset, AssetRef) else AssetRef(kind="depth", path=Path(path_or_asset))
        depth = self.load_asset(asset)
        if not isinstance(depth, torch.Tensor):
            raise TypeError(f"Decoded depth loader returned {type(depth).__name__}")
        if target_size is None:
            return depth
        return resize_depth_tensor(depth, target_size)

    def load_confidence(
        self,
        path_or_asset: Path | AssetRef,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        asset = (
            path_or_asset
            if isinstance(path_or_asset, AssetRef)
            else AssetRef(kind="confidence", path=Path(path_or_asset))
        )
        confidence = self.load_asset(asset)
        if not isinstance(confidence, torch.Tensor):
            raise TypeError(f"Decoded confidence loader returned {type(confidence).__name__}")
        if target_size is None:
            return confidence
        return resize_depth_tensor(confidence, target_size)

    def resize_alpha(self, alpha: torch.Tensor | None, target_size: tuple[int, int]) -> torch.Tensor | None:
        return resize_alpha_tensor(alpha, target_size)


LocalDataStore = LocalDataStoreV2
