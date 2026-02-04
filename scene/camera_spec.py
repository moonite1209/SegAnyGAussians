# scene/camera_spec.py
"""
相机元数据定义。

相机参数被分离为两个独立的组件：
- CameraParams: 渲染参数（内参、外参），仅用于渲染
- CameraMetaData: 数据文件路径，仅用于训练时加载数据

设计原则:
- frozen=True: 防止意外修改
- CameraParams 只包含渲染所需的轻量级数据
- CameraMetaData 只包含文件路径
- 支持快速序列化和反序列化
- 预计算变换矩阵并缓存（在 CameraParams 中）
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import math


@dataclass(frozen=True)
class CameraParams:
    """
    相机渲染参数（内参和外参）。

    包含 ONLY 3D Gaussian Splatting 渲染所需的参数。
    没有数据路径，没有加载相关的关注点。

    Attributes:
        uid: 相机唯一标识符
        width: 图像宽度（像素）
        height: 图像高度（像素）
        fx: X方向焦距（像素）
        fy: Y方向焦距（像素）
        cx: X方向主点（像素）
        cy: Y方向主点（像素）
        R: 旋转矩阵 (3, 3)
        T: 平移向量 (3,)
    """
    # 基础标识
    uid: int

    # 相机内参
    width: int
    height: int
    fx: float      # 焦距x
    fy: float      # 焦距y
    cx: float      # 主点x
    cy: float      # 主点y

    # 相机外参
    R: np.ndarray  # 旋转矩阵 (3,3)
    T: np.ndarray  # 平移向量 (3,)

    def __post_init__(self):
        """确保numpy数组是只读的，并验证参数。"""
        # 确保数组只读
        object.__setattr__(self, 'R', self.R.view())
        self.R.setflags(write=False)
        object.__setattr__(self, 'T', self.T.view())
        self.T.setflags(write=False)

        # 验证参数
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Invalid focal length: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")

    @property
    def FoVx(self) -> float:
        """从焦距计算X方向FoV（Field of View，弧度）"""
        return 2 * math.atan(self.width / (2 * self.fx))

    @property
    def FoVy(self) -> float:
        """从焦距计算Y方向FoV（Field of View，弧度）"""
        return 2 * math.atan(self.height / (2 * self.fy))

    def get_world_view_transform(self, trans: np.ndarray = np.zeros(3), scale: float = 1.0) -> np.ndarray:
        """
        获取世界到视图变换矩阵（带缓存）。

        Args:
            trans: 平移向量 (3,)
            scale: 缩放因子

        Returns:
            4x4世界到视图变换矩阵
        """
        # 使用对象存储缓存（绕过frozen限制）
        if not hasattr(self, '_cached_transforms'):
            object.__setattr__(self, '_cached_transforms', {})

        cache_key = (trans.tobytes(), scale)
        if cache_key not in self._cached_transforms:
            from utils.graphics_utils import getWorld2View2
            self._cached_transforms[cache_key] = getWorld2View2(
                self.R, self.T, trans, scale
            )
        return self._cached_transforms[cache_key]

    def get_projection_matrix(self, znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
        """
        获取投影矩阵（带缓存）。

        Args:
            znear: 近裁剪面距离
            zfar: 远裁剪面距离

        Returns:
            4x4投影矩阵
        """
        # 使用对象存储缓存（绕过frozen限制）
        if not hasattr(self, '_cached_transforms'):
            object.__setattr__(self, '_cached_transforms', {})

        cache_key = ('projection', znear, zfar)
        if cache_key not in self._cached_transforms:
            from utils.graphics_utils import getProjectionMatrix
            self._cached_transforms[cache_key] = getProjectionMatrix(
                znear, zfar, self.FoVx, self.FoVy
            )
        return self._cached_transforms[cache_key]


@dataclass(frozen=True)
class CameraMetaData:
    """
    数据文件路径，用于相机数据加载。

    包含 ONLY 加载训练数据所需的路径。
    没有渲染参数，没有变换相关的关注点。

    Attributes:
        image_name: 图像名称（不含扩展名）
        image_path: 图像文件路径
        mask_path: Mask文件路径（可选）
        labels_path: 标签文件路径（可选）
        depth_path: 深度文件路径（可选）
        confidence_path: 置信度文件路径（可选）
    """
    image_name: str
    image_path: Path
    mask_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    confidence_path: Optional[Path] = None

    def __post_init__(self):
        """确保所有路径都是 Path 对象。"""
        if not isinstance(self.image_path, Path):
            object.__setattr__(self, 'image_path', Path(self.image_path))

        # 可选路径
        for field in ['mask_path', 'labels_path', 'depth_path', 'confidence_path']:
            path = getattr(self, field)
            if path is not None and not isinstance(path, Path):
                object.__setattr__(self, field, Path(path))


def from_camera_info(cam_info: 'CameraInfo') -> Tuple[CameraParams, CameraMetaData]:
    """
    从现有的 CameraInfo 创建 CameraParams 和 CameraMetaData。

    Args:
        cam_info: CameraInfo对象（来自scene.dataset_readers）

    Returns:
        (CameraParams, CameraMetaData) 元组
    """
    from utils.graphics_utils import fov2focal

    # 计算焦距从FoV
    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)

    # 创建渲染参数
    params = CameraParams(
        uid=cam_info.uid,
        width=cam_info.width,
        height=cam_info.height,
        fx=fx,
        fy=fy,
        cx=cam_info.cx if cam_info.cx is not None else cam_info.width / 2,
        cy=cam_info.cy if cam_info.cy is not None else cam_info.height / 2,
        R=cam_info.R,
        T=cam_info.T,
    )

    # 创建数据元数据
    metadata = CameraMetaData(
        image_name=cam_info.image_name,
        image_path=Path(cam_info.image_path),
        mask_path=Path(cam_info.masks_path) if cam_info.masks_path else None,
        labels_path=Path(cam_info.labels_path) if cam_info.labels_path else None,
        depth_path=Path(cam_info.depth_path) if cam_info.depth_path else None,
        confidence_path=Path(cam_info.confidence_path) if cam_info.confidence_path else None,
    )

    return params, metadata


@dataclass(frozen=True)
class CameraSpec:
    """
    向后兼容的组合类：CameraParams + CameraMetaData。

    This class is kept for backward compatibility with existing code.
    New code should use CameraParams and CameraMetaData separately.

    Attributes:
        params: CameraParams containing rendering parameters
        metadata: CameraMetaData containing data file paths
    """
    params: CameraParams
    metadata: CameraMetaData

    # Convenience properties (delegate to components)
    @property
    def uid(self) -> int: return self.params.uid
    @property
    def image_name(self) -> str: return self.metadata.image_name
    @property
    def width(self) -> int: return self.params.width
    @property
    def height(self) -> int: return self.params.height
    @property
    def fx(self) -> float: return self.params.fx
    @property
    def fy(self) -> float: return self.params.fy
    @property
    def cx(self) -> float: return self.params.cx
    @property
    def cy(self) -> float: return self.params.cy
    @property
    def R(self) -> np.ndarray: return self.params.R
    @property
    def T(self) -> np.ndarray: return self.params.T
    @property
    def FoVx(self) -> float: return self.params.FoVx
    @property
    def FoVy(self) -> float: return self.params.FoVy
    @property
    def image_path(self) -> Path: return self.metadata.image_path
    @property
    def mask_path(self) -> Optional[Path]: return self.metadata.mask_path
    @property
    def labels_path(self) -> Optional[Path]: return self.metadata.labels_path
    @property
    def depth_path(self) -> Optional[Path]: return self.metadata.depth_path
    @property
    def confidence_path(self) -> Optional[Path]: return self.metadata.confidence_path

    def get_world_view_transform(self, trans: np.ndarray = np.zeros(3), scale: float = 1.0) -> np.ndarray:
        return self.params.get_world_view_transform(trans, scale)

    def get_projection_matrix(self, znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
        return self.params.get_projection_matrix(znear, zfar)

    @classmethod
    def from_camera_info(cls, cam_info: 'CameraInfo') -> 'CameraSpec':
        """
        从现有的CameraInfo创建CameraSpec（向后兼容）。

        Args:
            cam_info: CameraInfo对象

        Returns:
            CameraSpec实例
        """
        params, metadata = from_camera_info(cam_info)
        return cls(params=params, metadata=metadata)
