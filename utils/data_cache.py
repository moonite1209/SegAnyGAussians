# utils/data_cache.py
"""
数据缓存管理。

提供智能缓存功能，避免重复处理数据。
"""

import torch
import hashlib
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union
import logging

logger = logging.getLogger(__name__)

# TYPE_CHECKING用于类型提示，避免循环导入
if TYPE_CHECKING:
    from scene.camera_spec import CameraSpec, CameraMetaData
    from scene.camera_data import CameraData


class DataCache:
    """
    数据缓存管理器，避免重复处理。

    功能:
    - 缓存处理后的数据（如调整大小后的图像）
    - 基于文件修改时间的失效策略
    - 支持手动失效
    - 自动清理损坏的缓存

    Attributes:
        cache_dir: 缓存目录路径
        enabled: 是否启用缓存
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        初始化数据缓存。

        Args:
            cache_dir: 缓存目录路径，默认为.cache/saga
            enabled: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/saga")
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"DataCache initialized with directory: {self.cache_dir}")
        else:
            logger.debug("DataCache is disabled")

    def _get_cache_key_from_metadata(
        self,
        metadata: 'CameraMetaData',
        resolution_scale: float
    ) -> str:
        """
        生成缓存键（从 CameraMetaData）。

        使用文件路径、修改时间和分辨率生成唯一键。

        Args:
            metadata: CameraMetaData实例
            resolution_scale: 分辨率缩放因子

        Returns:
            MD5哈希字符串
        """
        try:
            image_stat = metadata.image_path.stat()
            key_data = f"{metadata.image_path}:{image_stat.st_mtime}:{image_stat.st_size}:{resolution_scale}"
        except Exception as e:
            # 如果无法获取文件状态，使用路径作为fallback
            logger.warning(f"Failed to get file stat for caching: {e}")
            key_data = f"{metadata.image_path}:{resolution_scale}"

        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_key_from_spec(
        self,
        spec: 'CameraSpec',
        resolution_scale: float
    ) -> str:
        """
        生成缓存键（从 CameraSpec，向后兼容）。

        使用文件路径、修改时间和分辨率生成唯一键。

        Args:
            spec: CameraSpec实例
            resolution_scale: 分辨率缩放因子

        Returns:
            MD5哈希字符串
        """
        try:
            image_stat = spec.image_path.stat()
            key_data = f"{spec.image_path}:{image_stat.st_mtime}:{image_stat.st_size}:{resolution_scale}"
        except Exception as e:
            # 如果无法获取文件状态，使用路径作为fallback
            logger.warning(f"Failed to get file stat for caching: {e}")
            key_data = f"{spec.image_path}:{resolution_scale}"

        return hashlib.md5(key_data.encode()).hexdigest()

    def get(
        self,
        spec_or_metadata: Union['CameraSpec', 'CameraMetaData'],
        resolution_scale: float
    ) -> Optional['CameraData']:
        """
        尝试从缓存获取数据。

        Args:
            spec_or_metadata: CameraSpec或CameraMetaData实例
            resolution_scale: 分辨率缩放因子

        Returns:
            CameraData实例，如果缓存未命中则返回None
        """
        if not self.enabled:
            return None

        # Determine cache key based on input type
        if hasattr(spec_or_metadata, 'image_name'):
            # Old CameraSpec (has image_name attribute)
            cache_key = self._get_cache_key_from_spec(spec_or_metadata, resolution_scale)
            name = spec_or_metadata.image_name
        else:
            # New CameraMetaData (no image_name attribute, has image_path)
            cache_key = self._get_cache_key_from_metadata(spec_or_metadata, resolution_scale)
            name = spec_or_metadata.image_path.stem

        cache_path = self.cache_dir / f"{cache_key}.pt"

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path)
            logger.debug(f"Cache hit: {name}")
            return data
        except Exception as e:
            # 缓存损坏，删除并返回None
            logger.warning(f"Corrupted cache file {cache_path}: {e}")
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def put(
        self,
        spec_or_metadata: Union['CameraSpec', 'CameraMetaData'],
        resolution_scale: float,
        data: 'CameraData'
    ):
        """
        将数据放入缓存。

        Args:
            spec_or_metadata: CameraSpec或CameraMetaData实例
            resolution_scale: 分辨率缩放因子
            data: CameraData实例
        """
        if not self.enabled:
            return

        # Determine cache key based on input type
        if hasattr(spec_or_metadata, 'image_name'):
            # Old CameraSpec (has image_name attribute)
            cache_key = self._get_cache_key_from_spec(spec_or_metadata, resolution_scale)
            name = spec_or_metadata.image_name
        else:
            # New CameraMetaData (no image_name attribute, has image_path)
            cache_key = self._get_cache_key_from_metadata(spec_or_metadata, resolution_scale)
            name = spec_or_metadata.image_path.stem

        cache_path = self.cache_dir / f"{cache_key}.pt"

        try:
            torch.save(data, cache_path)
            logger.debug(f"Cached: {name}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {name}: {e}")

    def invalidate(self, spec_or_metadata: Optional[Union['CameraSpec', 'CameraMetaData']] = None):
        """
        失效缓存。

        Args:
            spec_or_metadata: CameraSpec或CameraMetaData实例，如果为None则清空所有缓存
        """
        if not self.enabled:
            return

        if spec_or_metadata is None:
            # 清空所有缓存
            count = 0
            for cache_file in self.cache_dir.glob("*.pt"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            logger.info(f"Cleared {count} cache files")
        else:
            # 失效特定相机的缓存（需要尝试不同的resolution_scale）
            count = 0
            name = (spec_or_metadata.image_name
                    if hasattr(spec_or_metadata, 'image_name')
                    else spec_or_metadata.image_path.stem)

            for resolution_scale in [1.0, 2.0, 4.0, 8.0]:
                if hasattr(spec_or_metadata, 'image_name'):
                    # Old CameraSpec
                    cache_key = self._get_cache_key_from_spec(spec_or_metadata, resolution_scale)
                else:
                    # New CameraMetaData
                    cache_key = self._get_cache_key_from_metadata(spec_or_metadata, resolution_scale)

                cache_path = self.cache_dir / f"{cache_key}.pt"
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_path}: {e}")
            logger.info(f"Invalidated {count} cache entries for {name}")

    def get_cache_size(self) -> int:
        """
        获取缓存大小（字节）。

        Returns:
            缓存总大小（字节）
        """
        if not self.enabled:
            return 0

        total = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                total += cache_file.stat().st_size
            except Exception:
                pass
        return total

    def get_cache_info(self) -> dict:
        """
        获取缓存统计信息。

        Returns:
            包含缓存统计信息的字典
        """
        if not self.enabled:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.pt"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
        }

    def cleanup_old_cache(self, max_age_days: int = 7, max_size_mb: float = 1024):
        """
        清理旧缓存。

        Args:
            max_age_days: 最大缓存年龄（天）
            max_size_mb: 最大缓存大小（MB）
        """
        if not self.enabled:
            return

        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        max_size_bytes = max_size_mb * 1024 * 1024

        # 按年龄排序
        cache_files = []
        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                mtime = cache_file.stat().st_mtime
                age = current_time - mtime
                size = cache_file.stat().st_size
                cache_files.append((cache_file, age, size))
            except Exception:
                pass

        cache_files.sort(key=lambda x: x[1])  # 按年龄排序

        # 删除过期的缓存
        removed_count = 0
        removed_size = 0
        total_size = sum(f[2] for f in cache_files)

        for cache_file, age, size in cache_files:
            # 删除条件：超过年龄限制 或 总大小超过限制
            if age > max_age_seconds or total_size > max_size_bytes:
                try:
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += size
                    total_size -= size
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} cache files ({removed_size / (1024*1024):.2f} MB)")
        else:
            logger.debug("No cache files needed cleanup")

    def __repr__(self) -> str:
        """字符串表示"""
        if not self.enabled:
            return "DataCache(disabled)"
        info = self.get_cache_info()
        return f"DataCache(enabled={self.enabled}, files={info['num_files']}, size={info['size_mb']:.2f}MB)"
