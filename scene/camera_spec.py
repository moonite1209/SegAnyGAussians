# scene/camera_spec.py
"""
相机元数据定义。

CameraSpec是一个不可变的数据类，包含所有不依赖图像数据的相机参数。
设计原则:
- frozen=True: 防止意外修改
- 只包含轻量级数据（不加载实际图像）
- 支持快速序列化和反序列化
- 预计算变换矩阵并缓存
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np
import math


@dataclass(frozen=True)
class CameraSpec:
    """
    不可变的相机元数据，包含所有不依赖图像数据的参数。

    Attributes:
        uid: 相机唯一标识符
        image_name: 图像名称（不含扩展名）
        width: 图像宽度（像素）
        height: 图像高度（像素）
        fx: X方向焦距（像素）
        fy: Y方向焦距（像素）
        cx: X方向主点（像素）
        cy: Y方向主点（像素）
        R: 旋转矩阵 (3, 3)
        T: 平移向量 (3,)
        image_path: 图像文件路径
        mask_path: Mask文件路径（可选）
        labels_path: 标签文件路径（可选）
        depth_path: 深度文件路径（可选）
        confidence_path: 置信度文件路径（可选）
    """
    # 基础标识
    uid: int
    image_name: str

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

    # 数据路径
    image_path: Path
    mask_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    confidence_path: Optional[Path] = None

    def __post_init__(self):
        """确保numpy数组是只读的"""
        object.__setattr__(self, 'R', self.R.view())
        self.R.setflags(write=False)
        object.__setattr__(self, 'T', self.T.view())
        self.T.setflags(write=False)

    @property
    def FoVx(self) -> float:
        """从焦距计算X方向FoV（Field of View）"""
        return 2 * math.atan(self.width / (2 * self.fx))

    @property
    def FoVy(self) -> float:
        """从焦距计算Y方向FoV（Field of View）"""
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

    @classmethod
    def from_camera_info(cls, cam_info: 'CameraInfo') -> 'CameraSpec':
        """
        从现有的CameraInfo创建CameraSpec。

        Args:
            cam_info: CameraInfo对象（来自scene.dataset_readers）

        Returns:
            CameraSpec实例
        """
        from utils.graphics_utils import fov2focal

        # 计算焦距从FoV
        fx = fov2focal(cam_info.FovX, cam_info.width)
        fy = fov2focal(cam_info.FovY, cam_info.height)

        return cls(
            uid=cam_info.uid,
            image_name=cam_info.image_name,
            width=cam_info.width,
            height=cam_info.height,
            fx=fx,
            fy=fy,
            cx=cam_info.cx if cam_info.cx is not None else cam_info.width / 2,
            cy=cam_info.cy if cam_info.cy is not None else cam_info.height / 2,
            R=cam_info.R,
            T=cam_info.T,
            image_path=Path(cam_info.image_path),
            mask_path=Path(cam_info.masks_path) if cam_info.masks_path else None,
            labels_path=Path(cam_info.labels_path) if cam_info.labels_path else None,
            depth_path=Path(cam_info.depth_path) if cam_info.depth_path else None,
            confidence_path=Path(cam_info.confidence_path) if cam_info.confidence_path else None,
        )
