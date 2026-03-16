from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from utils.general_utils import PILtoTorch
from .dmb import read_dmb_file


class LocalDataStore:
    """
    Local filesystem datastore with per-worker LRU cache for decoded images.
    """

    def __init__(self, max_cache_bytes: int = 512 * 1024 * 1024):
        """
        Args:
            max_cache_bytes: Maximum cache size in bytes (default 512MB).
                             Set to 0 or negative to disable caching.
        """
        self._max_cache_bytes = max_cache_bytes
        self._cache: "OrderedDict[str, Tuple[torch.Tensor, Optional[torch.Tensor], int]]" = OrderedDict()
        self._cache_bytes = 0
        self._label_features_cache: dict[str, torch.Tensor] = {}

    def _cache_key(self, path: Path, target_size: Tuple[int, int]) -> str:
        return f"{str(path)}::{target_size[0]}x{target_size[1]}"

    def _cache_put(self, key: str, image: torch.Tensor, alpha: Optional[torch.Tensor]):
        if self._max_cache_bytes <= 0:
            return

        size = image.element_size() * image.nelement()
        if alpha is not None:
            size += alpha.element_size() * alpha.nelement()

        if size > self._max_cache_bytes:
            return

        while self._cache_bytes + size > self._max_cache_bytes and self._cache:
            _, (_, _, evict_size) = self._cache.popitem(last=False)
            self._cache_bytes -= evict_size

        self._cache[key] = (image, alpha, size)
        self._cache_bytes += size

    def _cache_get(self, key: str):
        if key not in self._cache:
            return None
        image, alpha, size = self._cache.pop(key)
        self._cache[key] = (image, alpha, size)
        return image, alpha

    def load_image(self, path: Path, target_size: Tuple[int, int]):
        key = self._cache_key(path, target_size)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        image = Image.open(path)
        resized = PILtoTorch(image, target_size)
        rgb = resized[:3, ...]
        alpha = resized[3:4, ...] if resized.shape[0] == 4 else None
        self._cache_put(key, rgb, alpha)
        return rgb, alpha

    def load_masks(self, path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            target_size: (target_w, target_h)
        """
        masks = torch.load(path, weights_only=True)
        target_w, target_h = target_size

        if masks.shape[0] == 0:
            return torch.empty((0, target_h, target_w), dtype=torch.bool)

        masks_float = masks.float()
        resized_masks_float = F.interpolate(
            masks_float.unsqueeze(1),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1)
        return (resized_masks_float > 0.5).bool()

    def load_label_features(self, path: Path) -> torch.Tensor:
        key = str(path)
        cached = self._label_features_cache.get(key)
        if cached is not None:
            return cached
        label_features = torch.load(path, weights_only=True)
        self._label_features_cache[key] = label_features
        return label_features

    def load_labels(self, path: Path, label_features_path: Path):
        labels = torch.load(path, weights_only=True)
        label_features = self.load_label_features(label_features_path)
        return labels, label_features

    def load_depth(self, path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            target_size: (target_w, target_h)
        """
        depth_array = read_dmb_file(path, is_confidence=False)
        depth_map = torch.from_numpy(depth_array.copy())[None, ...]
        target_w, target_h = target_size
        resized = F.interpolate(
            depth_map.unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0)
        return resized

    def load_confidence(self, path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            target_size: (target_w, target_h)
        """
        conf_array = read_dmb_file(path, is_confidence=True)
        conf_map = torch.from_numpy(conf_array.copy())[None, ...]
        target_w, target_h = target_size
        resized = F.interpolate(
            conf_map.unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0)
        return resized
