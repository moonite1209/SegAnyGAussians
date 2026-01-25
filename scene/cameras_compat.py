"""
Backward compatibility layer for camera loading.

This module provides compatibility functions that allow existing code to work
with the new Camera architecture without modification.

The key functions are:
- loadCam(): Load a Camera from CameraInfo (compatible with old signature)
- cameraList_from_camInfos(): Load a list of cameras from CameraInfo list

Internally, these use the new CameraSpec + CameraLoader architecture.
"""

from typing import List
from tqdm import tqdm

from scene.camera_new import Camera
from scene.camera_spec import CameraSpec
from scene.camera_data import CameraData
from utils.camera_loader import CameraLoader


def create_legacy_camera(
    cam_info,
    resolution: int,
    idx: int,
    resolution_scale: float = 1.0,
    data_device: str = "cuda"
) -> Camera:
    """
    Create a Camera object from CameraInfo (using new architecture internally).

    This is the recommended way to create a Camera from CameraInfo with the new architecture.
    It loads all data immediately (non-lazy mode) for backward compatibility.

    Args:
        cam_info: CameraInfo object with metadata and paths
        resolution: Target resolution (1, 2, 4, 8, -1, or custom)
        idx: Camera index/unique ID
        resolution_scale: Additional resolution scale factor
        data_device: Device to load data onto ('cuda' or 'cpu')

    Returns:
        Camera object with all data loaded
    """
    # 1. Create CameraSpec from CameraInfo
    spec = CameraSpec.from_camera_info(cam_info)

    # 2. Load data using CameraLoader
    loader = CameraLoader(resolution_scale=resolution_scale, resolution=resolution)
    data = loader.load(spec)

    # 3. Create Camera with pre-loaded data (non-lazy mode)
    trans = cam_info.trans if hasattr(cam_info, 'trans') else np.array([0.0, 0.0, 0.0])
    scale = cam_info.scale if hasattr(cam_info, 'scale') else 1.0

    camera = Camera(
        spec,
        data=data,
        lazy_load=False,  # Data is already loaded
        trans=trans,
        scale=scale
    )

    # 4. Move to specified device
    camera.to(data_device)

    return camera


def loadCam(resolution: int, id: int, cam_info, resolution_scale: float = 1.0) -> Camera:
    """
    Load a single camera from CameraInfo (backward compatible).

    This function maintains the same signature as the original loadCam() from utils/camera_utils.py
    but internally uses the new CameraSpec + CameraLoader architecture.

    Args:
        resolution: Target resolution (1, 2, 4, 8, -1, or custom)
        id: Camera ID/index
        cam_info: CameraInfo object with metadata and paths
        resolution_scale: Additional resolution scale factor

    Returns:
        Camera object with all data loaded

    Examples:
        >>> camera = loadCam(1, 0, cam_info, resolution_scale=1.0)
        >>> image = camera.original_image
    """
    return create_legacy_camera(cam_info, resolution, id, resolution_scale)


def cameraList_from_camInfos(
    cam_infos: List,
    resolution_scale: float,
    resolution: int,
    verbose: bool = True
) -> List[Camera]:
    """
    Load a list of cameras from CameraInfo list (backward compatible).

    This function maintains the same signature as the original cameraList_from_camInfos()
    from utils/camera_utils.py but internally uses the new architecture.

    Args:
        cam_infos: List of CameraInfo objects
        resolution_scale: Resolution scale factor
        resolution: Target resolution
        verbose: Whether to show progress bar

    Returns:
        List of Camera objects

    Examples:
        >>> cameras = cameraList_from_camInfos(cam_infos, 1.0, 1)
        >>> for camera in cameras:
        ...     print(camera.image_name, camera.original_image.shape)
    """
    camera_list = []
    iterator = tqdm(cam_infos, desc="Loading cameras") if verbose else cam_infos

    for idx, cam_info in enumerate(iterator):
        camera_list.append(loadCam(resolution, idx, cam_info, resolution_scale))

    return camera_list


# Preserve legacy Camera and MiniCamera imports for backward compatibility
# These import from the ORIGINAL cameras.py (not camera_new.py)
from scene.cameras import Camera as LegacyCamera
from scene.cameras import MiniCamera

# Import new CameraDataset for easier access
from scene.camera_dataset_new import CameraDataset, cameraDataset_from_camInfos


# Export both new and legacy for gradual migration
__all__ = [
    'Camera',  # New Camera class
    'CameraDataset',  # New CameraDataset class
    'loadCam',  # Compatible load function
    'cameraList_from_camInfos',  # Compatible list loader
    'cameraDataset_from_camInfos',  # New dataset creation function
    'create_legacy_camera',  # New recommended function
    'LegacyCamera',  # Old Camera class (for reference)
    'MiniCamera',  # MiniCamera from original code
]
