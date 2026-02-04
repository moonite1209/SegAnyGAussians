"""
Camera classes with inheritance-based separation.

This module provides:
- RenderCamera: Base class for rendering only (no data loading)
- TrainCamera: Extended class with lazy-loaded data access for training

Design principles:
- RenderCamera: Pure rendering, zero file system dependency
- TrainCamera: Inherits rendering, adds lazy-loaded data access
- Type safety: RenderCamera literally cannot access paths
- Memory management: Explicit load() and unload() methods
"""

from typing import Optional
import torch
import numpy as np

from scene.camera_spec import CameraParams, CameraMetaData
from scene.camera_data import CameraData


class RenderCamera:
    """
    Base camera class for rendering only (no data loading).

    This class contains ONLY rendering functionality - no file system access,
    no data loading, no paths. Suitable for GUI, inference, and pure rendering.

    Args:
        params: CameraParams containing rendering parameters (intrinsics + extrinsics)
        trans: Translation vector for coordinate normalization
        scale: Scale factor for coordinate normalization

    Examples:
        >>> params = CameraParams(uid=0, width=1920, height=1080, fx=1000, fy=1000, ...)
        >>> camera = RenderCamera(params)
        >>> # Use for rendering
        >>> viewpoint_cam = camera
    """

    def __init__(
        self,
        params: CameraParams,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0
    ):
        self._params = params
        self.trans = trans
        self.scale = scale

        # Near/far planes for projection
        self.zfar = 100.0
        self.znear = 0.01

        # Pre-compute transform matrices (cached on CPU, converted to GPU on demand)
        self._transform_cache = {}
        self._init_transforms()

    def _init_transforms(self):
        """Initialize transform matrices from params."""
        # Get world view transform (CPU numpy array)
        wv_transform = self._params.get_world_view_transform(self.trans, self.scale)
        self._world_view_transform_np = wv_transform

        # Get projection matrix (CPU numpy array)
        proj_matrix = self._params.get_projection_matrix(self.znear, self.zfar)
        self._projection_matrix_np = proj_matrix

    # ===== Params properties (delegate to CameraParams) =====

    @property
    def uid(self) -> int:
        """Camera unique identifier."""
        return self._params.uid

    @property
    def colmap_id(self) -> int:
        """COLMAP camera ID (alias for uid)."""
        return self._params.uid

    @property
    def image_width(self) -> int:
        """Image width in pixels."""
        return self._params.width

    @property
    def image_height(self) -> int:
        """Image height in pixels."""
        return self._params.height

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (3x3)."""
        return self._params.R

    @property
    def T(self) -> np.ndarray:
        """Translation vector (3,)."""
        return self._params.T

    @property
    def FoVx(self) -> float:
        """Field of view in X direction (radians)."""
        return self._params.FoVx

    @property
    def FoVy(self) -> float:
        """Field of view in Y direction (radians)."""
        return self._params.FoVy

    @property
    def cx(self) -> float:
        """Principal point X coordinate."""
        return self._params.cx

    @property
    def cy(self) -> float:
        """Principal point Y coordinate."""
        return self._params.cy

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

    def to(self, device: str) -> 'RenderCamera':
        """
        Move camera to specified device.

        This moves all transform matrices to the specified device.

        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')

        Returns:
            self (for method chaining)
        """
        for key in self._transform_cache:
            self._transform_cache[key] = self._transform_cache[key].to(device)
        return self

    # ===== Utility methods =====

    def __repr__(self) -> str:
        """String representation of the camera."""
        return f"RenderCamera(id={self.uid}, res={self.image_width}x{self.image_height})"


class TrainCamera(RenderCamera):
    """
    Extended camera class with lazy-loaded data access for training.

    Inherits all rendering functionality from RenderCamera and adds
    lazy-loaded data access via CameraMetaData.

    Args:
        params: CameraParams containing rendering parameters
        metadata: CameraMetaData containing data file paths
        trans: Translation vector for coordinate normalization
        scale: Scale factor for coordinate normalization

    Examples:
        >>> params, metadata = from_camera_info(cam_info)
        >>> camera = TrainCamera(params, metadata)
        >>> image = camera.original_image  # Triggers lazy load
        >>> camera.unload()  # Free memory when done
    """

    def __init__(
        self,
        params: CameraParams,
        metadata: CameraMetaData,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0
    ):
        # Initialize base with params (rendering only)
        super().__init__(params, trans, scale)

        # Store metadata for data loading
        self._metadata = metadata
        self._data = CameraData()  # Empty, will be loaded on demand

    # ===== Metadata properties (delegate to CameraMetaData) =====

    @property
    def image_name(self) -> str:
        """Image filename."""
        return self._metadata.image_name

    # ===== Lazy loading support =====

    def _ensure_loaded(self):
        """Ensure data is loaded (internal method)."""
        if not self._data.is_loaded:
            from utils.camera_loader import CameraLoader
            loader = CameraLoader(resolution_scale=1.0)
            # Load from params and metadata
            self._data = loader._load_from_params_and_metadata(self._params, self._metadata)

    def load(self, resolution_scale: float = 1.0):
        """
        Load camera data from disk.

        This method loads all camera data (image, masks, labels, depth, etc.)
        from the paths specified in the CameraMetaData.

        Args:
            resolution_scale: Scale factor for resolution (1.0 = full resolution)
        """
        if self._data.is_loaded:
            return  # Already loaded

        from utils.camera_loader import CameraLoader
        loader = CameraLoader(resolution_scale)
        self._data = loader._load_from_params_and_metadata(self._params, self._metadata)
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

    # ===== Device management =====

    def to(self, device: str) -> 'TrainCamera':
        """
        Move camera to specified device.

        This moves all transform matrices and loaded data to the specified device.

        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')

        Returns:
            self (for method chaining)
        """
        super().to(device)
        if self._data.is_loaded:
            self._data.to(device)
        return self

    # ===== Utility methods =====

    def __repr__(self) -> str:
        """String representation of the camera."""
        loaded_str = "loaded" if self._data.is_loaded else "unloaded"
        return (f"TrainCamera(id={self.uid}, name={self.image_name}, "
                f"res={self.image_width}x{self.image_height}, data={loaded_str})")


# Backward compatibility: alias for old Camera class
Camera = TrainCamera
