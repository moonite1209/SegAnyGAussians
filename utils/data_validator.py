# utils/data_validator.py
"""
数据验证工具。

提供多层次的数据验证功能，检查数据完整性和格式。
"""

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
import torch

# TYPE_CHECKING用于类型提示，避免循环导入
if TYPE_CHECKING:
    from scene.camera_spec import CameraSpec
    from scene.camera_data import CameraData


@dataclass
class ValidationResult:
    """
    验证结果。

    Attributes:
        is_valid: 验证是否通过
        errors: 错误列表
        warnings: 警告列表
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        """
        添加错误信息。

        Args:
            message: 错误消息
        """
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """
        添加警告信息。

        Args:
            message: 警告消息
        """
        self.warnings.append(message)

    def raise_if_invalid(self):
        """如果验证失败，抛出异常"""
        if not self.is_valid:
            error_msg = "Data validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            if self.warnings:
                error_msg += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
            raise DataValidationError(error_msg)

    def __str__(self) -> str:
        """格式化输出验证结果"""
        lines = []
        if self.is_valid:
            lines.append("✓ Validation passed")
        else:
            lines.append("✗ Validation failed")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


class DataValidationError(Exception):
    """数据验证异常"""
    pass


class DataValidator:
    """
    数据验证器，检查数据完整性和格式。

    验证项:
    - 文件存在性
    - 文件格式
    - 数据维度
    - 数据范围
    - 相机参数合理性
    """

    def validate_scene_structure(self, source_path: Path) -> ValidationResult:
        """
        验证场景目录结构。

        Args:
            source_path: 场景根目录路径

        Returns:
            ValidationResult对象
        """
        result = ValidationResult(True, [], [])

        # 检查必需目录
        required_dirs = ['images', 'sparse']
        for dir_name in required_dirs:
            dir_path = source_path / dir_name
            if not dir_path.exists():
                result.add_error(f"Required directory missing: {dir_name}")
            elif not dir_path.is_dir():
                result.add_error(f"Path exists but is not a directory: {dir_name}")

        # 检查COLMAP文件
        sparse_path = source_path / 'sparse'
        if sparse_path.exists():
            required_files = ['images.bin', 'cameras.bin', 'points3D.bin']
            for file_name in required_files:
                bin_path = sparse_path / file_name
                txt_path = sparse_path / file_name.replace('.bin', '.txt')

                if not bin_path.exists() and not txt_path.exists():
                    # 尝试接受txt格式
                    result.add_warning(f"COLMAP file not found: {file_name} (tried both .bin and .txt)")

        # 检查图像目录
        images_path = source_path / 'images'
        if images_path.exists():
            # 检查是否有图像文件
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            if not image_files:
                result.add_error("No image files found in images/ directory (expected .jpg or .png)")
            else:
                result.add_warning(f"Found {len(image_files)} image files")
        else:
            result.add_error("images/ directory not found")

        return result

    def validate_camera_spec(self, spec: 'CameraSpec') -> ValidationResult:
        """
        验证CameraSpec。

        Args:
            spec: CameraSpec实例

        Returns:
            ValidationResult对象
        """
        result = ValidationResult(True, [], [])

        # 检查图像文件
        if not spec.image_path.exists():
            result.add_error(f"Image file not found: {spec.image_path}")
        else:
            # 检查文件是否可读
            try:
                with open(spec.image_path, 'rb') as f:
                    f.read(1)
            except Exception as e:
                result.add_error(f"Image file is not readable: {spec.image_path} ({str(e)})")

        # 检查分辨率
        if spec.width <= 0 or spec.height <= 0:
            result.add_error(f"Invalid resolution: {spec.width}x{spec.height}")

        # 检查图像是否过大（警告）
        if spec.width > 8000 or spec.height > 8000:
            result.add_warning(f"Very large image resolution: {spec.width}x{spec.height} (may cause memory issues)")

        # 检查焦距
        if spec.fx <= 0 or spec.fy <= 0:
            result.add_error(f"Invalid focal length: fx={spec.fx}, fy={spec.fy}")

        # 检查主点
        if spec.cx < 0 or spec.cx > spec.width:
            result.add_error(f"Invalid cx (principal point x): {spec.cx} (should be in [0, {spec.width}])")
        if spec.cy < 0 or spec.cy > spec.height:
            result.add_error(f"Invalid cy (principal point y): {spec.cy} (should be in [0, {spec.height}])")

        # 检查旋转矩阵
        if spec.R.shape != (3, 3):
            result.add_error(f"Invalid rotation matrix shape: {spec.R.shape} (expected (3, 3))")
        else:
            # 检查是否为有效的旋转矩阵
            # R^T * R = I
            rt_r = spec.R.T @ spec.R
            identity = np.eye(3)
            if not np.allclose(rt_r, identity, atol=1e-5):
                result.add_error("Rotation matrix is not orthogonal (R^T * R != I)")
            # det(R) = 1
            if not np.isclose(np.linalg.det(spec.R), 1.0, atol=1e-5):
                result.add_error(f"Rotation matrix determinant is not 1: {np.linalg.det(spec.R)}")

        # 检查平移向量
        if spec.T.shape != (3,):
            result.add_error(f"Invalid translation vector shape: {spec.T.shape} (expected (3,))")

        # 检查平移向量是否过大（警告）
        if np.any(np.abs(spec.T) > 1000):
            result.add_warning(f"Large translation values detected: {spec.T} (may indicate incorrect scale)")

        # 检查可选文件
        if spec.mask_path and not spec.mask_path.exists():
            result.add_warning(f"Mask file not found: {spec.mask_path}")
        if spec.labels_path and not spec.labels_path.exists():
            result.add_warning(f"Labels file not found: {spec.labels_path}")
        if spec.depth_path and not spec.depth_path.exists():
            result.add_warning(f"Depth file not found: {spec.depth_path}")
        if spec.confidence_path and not spec.confidence_path.exists():
            result.add_warning(f"Confidence file not found: {spec.confidence_path}")

        return result

    def validate_camera_data(self, data: 'CameraData', expected_shape: tuple) -> ValidationResult:
        """
        验证CameraData。

        Args:
            data: CameraData实例
            expected_shape: 期望的图像形状 (C, H, W)

        Returns:
            ValidationResult对象
        """
        result = ValidationResult(True, [], [])

        # 检查图像
        if data.image is None:
            result.add_error("Image data is None")
        else:
            # 检查数据类型
            if not isinstance(data.image, torch.Tensor):
                result.add_error(f"Image must be torch.Tensor, got {type(data.image)}")
            else:
                # 检查形状
                if data.image.ndim != 3:
                    result.add_error(f"Image must be 3D, got {data.image.ndim}D with shape {data.image.shape}")
                elif data.image.shape[0] not in [1, 3, 4]:
                    result.add_error(f"Image channels must be 1, 3, or 4, got {data.image.shape[0]}")
                else:
                    # 检查是否与期望形状匹配（只警告，因为可能有缩放）
                    if data.image.shape != expected_shape:
                        result.add_warning(f"Image shape differs from expected: expected {expected_shape}, got {data.image.shape}")

                    # 检查值范围
                    if data.image.min() < 0:
                        result.add_error(f"Image has negative values: min={data.image.min()}")
                    if data.image.max() > 1:
                        result.add_warning(f"Image values exceed 1.0 (expected range [0, 1]): max={data.image.max()}")

        # 检查alpha mask
        if data.alpha_mask is not None:
            if data.alpha_mask.ndim != 3 or data.alpha_mask.shape[0] != 1:
                result.add_error(f"Alpha mask must have shape (1, H, W), got {data.alpha_mask.shape}")

        # 检查masks
        if data.masks is not None:
            if data.masks.ndim != 3:
                result.add_error(f"Masks must be 3D, got shape {data.masks.shape}")
            elif data.masks.dtype != torch.bool:
                result.add_warning(f"Masks should be bool for efficiency, got {data.masks.dtype}")

        # 检查labels
        if data.labels is not None:
            if data.labels.ndim != 1:
                result.add_error(f"Labels must be 1D, got shape {data.labels.shape}")
            elif data.labels.dtype not in [torch.int32, torch.int64, torch.int8]:
                result.add_warning(f"Labels should be integer type, got {data.labels.dtype}")

        # 检查depth map
        if data.depth_map is not None:
            if data.depth_map.ndim != 2:
                result.add_error(f"Depth map must be 2D, got shape {data.depth_map.shape}")
            else:
                # 检查深度值的合理性
                if data.depth_map.min() < 0:
                    result.add_error(f"Depth map has negative values: min={data.depth_map.min()}")
                if data.depth_map.max() > 1000:  # 假设深度单位是米
                    result.add_warning(f"Depth map has very large values: max={data.depth_map.max()}")

        # 检查confidence map
        if data.confidence_map is not None:
            if data.confidence_map.ndim != 2:
                result.add_error(f"Confidence map must be 2D, got shape {data.confidence_map.shape}")
            else:
                # 置信度通常在[0, 1]或[0, 255]
                if data.confidence_map.max() > 1.0 and data.confidence_map.max() <= 255.0:
                    result.add_warning("Confidence map appears to be in [0, 255] range, expected [0, 1]")
                elif data.confidence_map.min() < 0 or data.confidence_map.max() > 1:
                    result.add_warning(f"Confidence map values out of [0, 1] range: [{data.confidence_map.min()}, {data.confidence_map.max()}]")

        return result
