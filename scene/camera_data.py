# scene/camera_data.py
"""
相机数据定义。

CameraData是一个可变的数据类，包含相机的所有图像数据。
支持懒加载和内存管理。
"""

from typing import Optional
from dataclasses import dataclass, field
import torch


@dataclass
class CameraData:
    """
    相机的图像数据，支持懒加载和内存管理。

    设计原则:
    - 可变数据类（支持数据加载和卸载）
    - 支持懒加载（按需加载）
    - 支持内存管理（主动卸载）
    - 明确的所有权（谁负责加载/卸载）

    Attributes:
        image: RGB图像张量 (3, H, W)，值范围[0, 1]
        alpha_mask: Alpha通道掩码 (1, H, W)，值范围[0, 1]
        masks: 分割掩码 (N, H, W)，bool类型
        labels: 类别标签 (N,)，int64类型
        label_features: 类别特征向量 (N, D)
        depth_map: 深度图 (H, W)
        confidence_map: 置信度图 (H, W)
        device: 当前所在设备
        _loaded: 数据是否已加载
        _resolution_scale: 分辨率缩放因子
    """
    # 图像数据
    image: Optional[torch.Tensor] = None  # (3, H, W)
    alpha_mask: Optional[torch.Tensor] = None  # (1, H, W)

    # 附加数据
    masks: Optional[torch.Tensor] = None  # (N, H, W) bool
    labels: Optional[torch.Tensor] = None  # (N,) int64
    label_features: Optional[torch.Tensor] = None  # (N, D)
    depth_map: Optional[torch.Tensor] = None  # (H, W)
    confidence_map: Optional[torch.Tensor] = None  # (H, W)

    # 元数据
    device: str = "cpu"
    _loaded: bool = False
    _resolution_scale: float = 1.0

    @property
    def is_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self._loaded

    def to(self, device: str) -> 'CameraData':
        """
        移动数据到指定设备。

        Args:
            device: 目标设备（如 'cuda', 'cpu'）

        Returns:
            self（支持链式调用）
        """
        self.device = device
        if self.image is not None:
            self.image = self.image.to(device)
        if self.alpha_mask is not None:
            self.alpha_mask = self.alpha_mask.to(device)
        if self.masks is not None:
            self.masks = self.masks.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.label_features is not None:
            self.label_features = self.label_features.to(device)
        if self.depth_map is not None:
            self.depth_map = self.depth_map.to(device)
        if self.confidence_map is not None:
            self.confidence_map = self.confidence_map.to(device)
        return self

    def unload(self):
        """
        卸载所有数据以释放内存。

        将所有tensor设置为None，释放GPU/CPU内存。
        """
        self.image = None
        self.alpha_mask = None
        self.masks = None
        self.labels = None
        self.label_features = None
        self.depth_map = None
        self.confidence_map = None
        self._loaded = False

    def __repr__(self) -> str:
        """字符串表示"""
        loaded_str = "loaded" if self._loaded else "unloaded"
        device_str = f"on {self.device}" if self._loaded else ""
        return f"CameraData({loaded_str} {device_str})"

    def get_memory_usage(self) -> dict:
        """
        获取当前内存使用情况。

        Returns:
            包含各tensor内存占用的字典（单位：字节）
        """
        usage = {}
        if self.image is not None:
            usage['image'] = self.image.element_size() * self.image.nelement()
        if self.alpha_mask is not None:
            usage['alpha_mask'] = self.alpha_mask.element_size() * self.alpha_mask.nelement()
        if self.masks is not None:
            usage['masks'] = self.masks.element_size() * self.masks.nelement()
        if self.labels is not None:
            usage['labels'] = self.labels.element_size() * self.labels.nelement()
        if self.label_features is not None:
            usage['label_features'] = self.label_features.element_size() * self.label_features.nelement()
        if self.depth_map is not None:
            usage['depth_map'] = self.depth_map.element_size() * self.depth_map.nelement()
        if self.confidence_map is not None:
            usage['confidence_map'] = self.confidence_map.element_size() * self.confidence_map.nelement()
        usage['total'] = sum(usage.values())
        return usage
