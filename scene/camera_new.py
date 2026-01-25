"""
Unified Camera class integrating metadata and data with lazy loading support.

This module provides the new Camera class that combines CameraSpec (immutable metadata)
and CameraData (mutable image data) with support for lazy loading and memory management.
"""

from typing import Optional
import torch
import numpy as np

from scene.camera_spec import CameraSpec
from scene.camera_data import CameraData
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:
    """
    Unified camera class integrating metadata and data.

    This class combines CameraSpec (immutable metadata) and CameraData (mutable image data)
    with support for lazy loading and explicit memory management.

    Design principles:
    - Separation of concerns: spec (immutable) vs data (mutable)
    - Lazy loading: data is loaded on-demand
    - Memory management: explicit load() and unload() methods
    - Backward compatibility: preserves all original Camera attributes

    Args:
        spec: CameraSpec containing immutable camera metadata
        data: Optional CameraData with pre-loaded image data
        lazy_load: If True, data is loaded on-demand; if False, data must be pre-loaded
        trans: Translation vector for coordinate normalization
        scale: Scale factor for coordinate normalization

    Examples:
        >>> # With lazy loading (recommended)
        >>> spec = CameraSpec.from_camera_info(cam_info)
        >>> camera = Camera(spec, lazy_load=True)
        >>> image = camera.original_image  # Data is loaded here
        >>>
        >>> # Without lazy loading (data pre-loaded)
        >>> loader = CameraLoader(resolution_scale=1.0)
        >>> data = loader.load(spec)
        >>> camera = Camera(spec, data, lazy_load=False)
        >>> image = camera.original_image  # Data is already loaded
    """

    def __init__(
        self,
        spec: CameraSpec,
        data: Optional[CameraData] = None,
        lazy_load: bool = True,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0
    ):
        self._spec = spec  # Immutable metadata
        self._data = data or CameraData()  # Mutable data
        self._lazy_load = lazy_load
        self.trans = trans
        self.scale = scale

        # Near/far planes for projection (must be set before _init_transforms)
        self.zfar = 100.0
        self.znear = 0.01

        # Pre-compute transform matrices (cached on CPU, converted to GPU on demand)
        self._transform_cache = {}
        self._init_transforms()

    def _init_transforms(self):
        """Initialize transform matrices from spec."""
        # Get world view transform (CPU numpy array)
        wv_transform = self._spec.get_world_view_transform(self.trans, self.scale)
        self._world_view_transform_np = wv_transform

        # Get projection matrix (CPU numpy array)
        proj_matrix = self._spec.get_projection_matrix(self.znear, self.zfar)
        self._projection_matrix_np = proj_matrix

    # ===== Legacy Camera attributes (backward compatibility) =====

    @property
    def uid(self) -> int:
        """Camera unique identifier."""
        return self._spec.uid

    @property
    def colmap_id(self) -> int:
        """COLMAP camera ID."""
        return self._spec.uid

    @property
    def image_name(self) -> str:
        """Image filename."""
        return self._spec.image_name

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (3x3)."""
        return self._spec.R

    @property
    def T(self) -> np.ndarray:
        """Translation vector (3,)."""
        return self._spec.T

    @property
    def FoVx(self) -> float:
        """Field of view in X direction (radians)."""
        return self._spec.FoVx

    @property
    def FoVy(self) -> float:
        """Field of view in Y direction (radians)."""
        return self._spec.FoVy

    @property
    def image_width(self) -> int:
        """Image width in pixels."""
        return self._spec.width

    @property
    def image_height(self) -> int:
        """Image height in pixels."""
        return self._spec.height

    @property
    def cx(self) -> Optional[float]:
        """Principal point X coordinate."""
        return self._spec.cx

    @property
    def cy(self) -> Optional[float]:
        """Principal point Y coordinate."""
        return self._spec.cy

    # ===== New API =====

    @property
    def spec(self) -> CameraSpec:
        """Get camera specification (read-only)."""
        return self._spec

    @property
    def data(self) -> CameraData:
        """Get camera data (mutable)."""
        return self._data

    # ===== Lazy loading support =====

    def _ensure_loaded(self):
        """Ensure data is loaded (internal method)."""
        if self._lazy_load and not self._data.is_loaded:
            self.load()

    def load(self, resolution_scale: float = 1.0):
        """
        Load camera data from disk.

        This method loads all camera data (image, masks, labels, depth, etc.)
        from the paths specified in the CameraSpec.

        Args:
            resolution_scale: Scale factor for resolution (1.0 = full resolution)
        """
        if self._data.is_loaded:
            return  # Already loaded

        from utils.camera_loader import CameraLoader
        loader = CameraLoader(resolution_scale)
        self._data = loader.load(self._spec)
        self._data._resolution_scale = resolution_scale

    def unload(self):
        """Unload camera data to release memory."""
        self._data.unload()

    # ===== Data access (auto-triggers lazy loading) =====

    @property
    def original_image(self) -> torch.Tensor:
        """
        Get original image tensor (lazy loading).

        Returns:
            Image tensor of shape (3, H, W) in [0, 1] range
        """
        self._ensure_loaded()
        return self._data.image

    @property
    def original_masks(self) -> Optional[torch.Tensor]:
        """
        Get segmentation masks (lazy loading).

        Returns:
            Masks tensor of shape (N, H, W) with dtype bool, or None
        """
        self._ensure_loaded()
        return self._data.masks

    @property
    def labels(self) -> Optional[torch.Tensor]:
        """
        Get class labels (lazy loading).

        Returns:
            Labels tensor of shape (N,) with dtype int64, or None
        """
        self._ensure_loaded()
        return self._data.labels

    @property
    def label_features(self) -> Optional[torch.Tensor]:
        """
        Get label features (lazy loading).

        Returns:
            Feature tensor of shape (N, D) with dtype float32, or None
        """
        self._ensure_loaded()
        return self._data.label_features

    @property
    def depth_map(self) -> Optional[torch.Tensor]:
        """
        Get depth map (lazy loading).

        Returns:
            Depth tensor of shape (1, H, W), or None
        """
        self._ensure_loaded()
        return self._data.depth_map

    @property
    def confidence_map(self) -> Optional[torch.Tensor]:
        """
        Get confidence map (lazy loading).

        Returns:
            Confidence tensor of shape (1, H, W), or None
        """
        self._ensure_loaded()
        return self._data.confidence_map

    # ===== Transform matrices (GPU tensors) =====

    @property
    def world_view_transform(self) -> torch.Tensor:
        """
        Get world-to-view transformation matrix (GPU tensor).

        Returns:
            4x4 transformation matrix as GPU tensor
        """
        if 'world_view_transform' not in self._transform_cache:
            matrix = torch.tensor(self._world_view_transform_np).transpose(0, 1)
            self._transform_cache['world_view_transform'] = matrix
        return self._transform_cache['world_view_transform']

    @property
    def projection_matrix(self) -> torch.Tensor:
        """
        Get projection matrix (GPU tensor).

        Returns:
            4x4 projection matrix as GPU tensor
        """
        if 'projection_matrix' not in self._transform_cache:
            matrix = torch.tensor(self._projection_matrix_np).transpose(0, 1)
            self._transform_cache['projection_matrix'] = matrix
        return self._transform_cache['projection_matrix']

    @property
    def full_proj_transform(self) -> torch.Tensor:
        """
        Get full projection transformation matrix (world to clip).

        Returns:
            4x4 combined transformation matrix as GPU tensor
        """
        return (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    @property
    def camera_center(self) -> torch.Tensor:
        """
        Get camera center in world coordinates.

        Returns:
            Camera center as (3,) tensor
        """
        return self.world_view_transform.inverse()[3, :3]

    # ===== Device management =====

    def to(self, device: str) -> 'Camera':
        """
        Move camera to specified device.

        This moves all transform matrices and loaded data to the specified device.

        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')

        Returns:
            self (for method chaining)
        """
        # Move transform matrices
        for key in self._transform_cache:
            self._transform_cache[key] = self._transform_cache[key].to(device)

        # Move data
        if self._data.is_loaded:
            self._data.to(device)

        return self

    # ===== Utility methods =====

    def __repr__(self) -> str:
        """String representation of the camera."""
        loaded_str = "loaded" if self._data.is_loaded else "unloaded"
        return (f"Camera(id={self.uid}, name={self.image_name}, "
                f"res={self.image_width}x{self.image_height}, data={loaded_str})")
