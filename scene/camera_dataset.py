"""
Enhanced CameraDataset with support for lazy loading, caching, and optimized data loading.

This module provides a new CameraDataset implementation that:
- Uses CameraParams and CameraMetaData for metadata (lightweight, separated)
- Supports lazy loading to reduce memory usage
- Integrates with DataCache for faster repeated access
- Works seamlessly with PyTorch DataLoader
"""

from typing import List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging

from scene.camera import TrainCamera
from scene.camera_spec import CameraParams, CameraMetaData, from_camera_info
from scene.camera_data import CameraData
from utils.camera_loader import CameraLoader, DataLoadError
from utils.data_cache import DataCache


logger = logging.getLogger(__name__)


class CameraDataset(Dataset):
    """
    Enhanced camera dataset with lazy loading and caching support.

    This dataset stores only (CameraParams, CameraMetaData) tuples (lightweight metadata) in memory,
    and loads actual image data on-demand. It supports:
    - Lazy loading: data is loaded when accessed
    - Caching: processed data is cached for faster repeated access
    - Memory management: explicit unload() to free memory

    Args:
        camera_tuples: List of (CameraParams, CameraMetaData) tuples
        resolution_scale: Scale factor for resolution (1.0 = full resolution)
        resolution: Target resolution (1, 2, 4, 8, -1, or custom)
        lazy_load: If True, load data on-demand; if False, load all data upfront
        use_cache: If True, enable disk caching of processed data
        cache_dir: Directory for cache files (default: .cache/saga)
        data_device: Device to load data onto ('cuda' or 'cpu')

    Examples:
        >>> tuples = [(params1, metadata1), (params2, metadata2), ...]
        >>> dataset = CameraDataset(camera_tuples, resolution_scale=1.0, resolution=1)
        >>> camera = dataset[0]  # Loads first camera
        >>> image = camera.original_image  # Access image data
    """

    def __init__(
        self,
        camera_tuples: List[Tuple[CameraParams, CameraMetaData]],
        resolution_scale: float = 1.0,
        resolution: int = 1,
        lazy_load: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        data_device: str = "cuda"
    ):
        self.camera_tuples = camera_tuples
        self.resolution_scale = resolution_scale
        self.resolution = resolution
        self.lazy_load = lazy_load
        self.data_device = data_device

        # Initialize loader
        self.loader = CameraLoader(resolution_scale=resolution_scale, resolution=resolution)

        # Initialize cache
        self.cache = DataCache(cache_dir, enabled=use_cache) if use_cache else None

        # Store loaded cameras (for non-lazy mode)
        self._loaded_cameras: dict[int, TrainCamera] = {}

        logger.info(
            f"Created CameraDataset with {len(camera_tuples)} cameras "
            f"(lazy_load={lazy_load}, use_cache={use_cache}, device={data_device})"
        )

    def __len__(self) -> int:
        """Return the number of cameras in the dataset."""
        return len(self.camera_tuples)

    def __getitem__(self, idx: int) -> TrainCamera:
        """
        Get a camera by index.

        This method implements the PyTorch Dataset interface. It:
        1. Checks if camera is already loaded (non-lazy mode)
        2. Tries to load from cache
        3. Loads from disk if not cached
        4. Moves data to specified device

        Args:
            idx: Camera index

        Returns:
            TrainCamera object with all metadata and data
        """
        # If already loaded (non-lazy mode), return directly
        if idx in self._loaded_cameras:
            return self._loaded_cameras[idx]

        params, metadata = self.camera_tuples[idx]

        # Try to load from cache
        if self.cache is not None:
            cached_data = self.cache.get(metadata, self.resolution_scale)
            if cached_data is not None:
                camera = TrainCamera(
                    params,
                    metadata,
                    data=cached_data
                )
                camera.to(self.data_device)

                if not self.lazy_load:
                    self._loaded_cameras[idx] = camera

                return camera

        # Load from disk
        try:
            data = self.loader._load_from_params_and_metadata(params, metadata)
        except DataLoadError as e:
            logger.error(f"Failed to load camera {idx}: {e}")
            raise

        # Put into cache
        if self.cache is not None:
            self.cache.put(metadata, self.resolution_scale, data)

        # Create camera
        camera = TrainCamera(
            params,
            metadata,
            data=data
        )
        camera.to(self.data_device)

        # Store in loaded cameras if non-lazy mode
        if not self.lazy_load:
            self._loaded_cameras[idx] = camera

        return camera

    def unload_all(self):
        """
        Unload all loaded cameras to free memory.

        This is useful for memory management during training.
        After calling this, subsequent accesses will re-load the data.
        """
        for camera in self._loaded_cameras.values():
            camera.unload()
        self._loaded_cameras.clear()

        logger.info("Unloaded all cameras from memory")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - enabled: Whether cache is enabled
            - size_bytes: Total cache size in bytes
            - size_mb: Total cache size in megabytes
        """
        if self.cache is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "size_bytes": self.cache.get_cache_size(),
            "size_mb": self.cache.get_cache_size() / (1024 * 1024)
        }

    def clear_cache(self):
        """Clear all cached data."""
        if self.cache is not None:
            self.cache.invalidate()
            logger.info("Cleared all cached data")


def cameraDataset_from_camInfos(
    cam_infos: List,
    resolution: int,
    resolution_scale: float = 1.0,
    lazy_load: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    data_device: str = "cuda"
) -> CameraDataset:
    """
    Create a CameraDataset from CameraInfo list (backward compatible).

    This function maintains compatibility with the original cameraDataset_from_camInfos
    while using the new architecture internally.

    Args:
        cam_infos: List of CameraInfo objects
        resolution: Target resolution (1, 2, 4, 8, -1, or custom)
        resolution_scale: Additional resolution scale factor
        lazy_load: If True, use lazy loading
        use_cache: If True, enable caching
        cache_dir: Cache directory
        data_device: Device to load data onto

    Returns:
        CameraDataset instance

    Examples:
        >>> dataset = cameraDataset_from_camInfos(cam_infos, 1, 1.0)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    """
    # Convert CameraInfo to (CameraParams, CameraMetaData) tuples
    camera_tuples = [from_camera_info(info) for info in cam_infos]

    # Create dataset
    dataset = CameraDataset(
        camera_tuples=camera_tuples,
        resolution_scale=resolution_scale,
        resolution=resolution,
        lazy_load=lazy_load,
        use_cache=use_cache,
        cache_dir=cache_dir,
        data_device=data_device
    )

    return dataset
