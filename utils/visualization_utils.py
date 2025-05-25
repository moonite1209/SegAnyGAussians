import sklearn.preprocessing
from sklearn.decomposition import PCA
from umap import UMAP
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm

def to_image(array, dataformat='HWC') -> Image.Image:
    if dataformat=='CHW':
        array=to_image_array(array, dataformat).transpose(1,2,0)
    else:
        array=to_image_array(array, dataformat)
    return Image.fromarray(array)

def to_image_array(array, dataformat='HWC', use_pca=False) -> np.ndarray:
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    else:
        raise TypeError()
    if dataformat=='CHW':
        array = array.transpose(1,2,0)
    else:
        pass

    h,w,c = array.shape
    array = (array*255).astype(np.uint8)

    if dataformat=='CHW':
        return array.transpose(2,0,1)
    else:
        return array

def save_image(array, path, dataformat='HWC'):
    image = to_image(array, dataformat)
    image.save(path)

def feature_map_to_image(feature_map, dataformat='HWC'):
    array = feature_map
    if isinstance(feature_map, np.ndarray):
        pass
    elif isinstance(feature_map, torch.Tensor):
        is_tensor = True
        device = feature_map.device
        feature_map = feature_map.detach().cpu().numpy()
    else:
        raise TypeError()
    if dataformat=='CHW':
        feature_map = feature_map.transpose(1,2,0)
    elif dataformat=='HW':
        feature_map = feature_map[...,None]
    else:
        pass
    
    h,w,c = feature_map.shape
    if c==1:
        feature_map_norm = cv2.normalize(feature_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        feature_map_rgb = cv2.applyColorMap(feature_map_norm, cv2.COLORMAP_JET)
    else:
        if c==2:
            feature_map = np.concat((feature_map, np.zeros((h,w,1))), axis=-1)
        feature_map_flat = feature_map.reshape(-1,c)
        pca = PCA(n_components=3)
        feature_map_3 = pca.fit_transform(feature_map_flat).reshape(h, w, -1)
        # umap = UMAP(n_components=3, metric='cosine')
        # feature_map_3 = umap.fit_transform(feature_map_flat).reshape(h, w, -1)
        feature_map_rgb = cv2.normalize(feature_map_3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    if dataformat=='CHW':
        feature_map_rgb = feature_map_rgb.transpose(2,0,1)
    return torch.from_numpy(feature_map_rgb).to(device) if is_tensor else feature_map_rgb

def hsv_to_rgb(hsv):
    """
    将 HSV 颜色 [N, 3] 转换为 RGB
    hsv[..., 0] ∈ [0,1] hue
    hsv[..., 1] ∈ [0,1] saturation
    hsv[..., 2] ∈ [0,1] value
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = (h * 6).floor() % 6
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i.to(torch.int32)

    rgb = torch.zeros_like(hsv)
    for j in range(6):
        mask = (i == j)
        if j == 0: rgb[mask] = torch.stack([v[mask], t[mask], p[mask]], dim=1)
        if j == 1: rgb[mask] = torch.stack([q[mask], v[mask], p[mask]], dim=1)
        if j == 2: rgb[mask] = torch.stack([p[mask], v[mask], t[mask]], dim=1)
        if j == 3: rgb[mask] = torch.stack([p[mask], q[mask], v[mask]], dim=1)
        if j == 4: rgb[mask] = torch.stack([t[mask], p[mask], v[mask]], dim=1)
        if j == 5: rgb[mask] = torch.stack([v[mask], p[mask], q[mask]], dim=1)

    return rgb

def features_to_color(features):
    """
    将 [N, C] 特征映射到 RGB 颜色。
    
    支持：
    - C == 1: 作为强度使用 colormap 直接映射
    - C == 2 or 3: 转换为 RGB (归一化)
    - C > 3: 使用 PCA 降到 3D 后归一化

    Args:
        features (torch.Tensor): 输入特征 [N, C]

    Returns:
        torch.Tensor: [N, 3] RGB 颜色张量，float32，范围 [0, 1]
    """
    if not torch.is_tensor(features):
        raise TypeError("features 应为 torch.Tensor")
    if features.dim() != 2:
        raise ValueError("features 应为 2D Tensor，形状 [N, C]")

    N, C = features.shape
    features = features.detach().cpu().float()

    if C == 1:
        # === 标量强度：调用 intensity_to_color ===
        x = features.squeeze(1)
        return intensity_to_color(x)

    elif C == 2:
        # === 将 2D 映射到 HSV 再转 RGB（保持空间结构） ===
        x = features
        # 标准化到 [0, 1]
        x -= x.min(dim=0, keepdim=True)[0]
        x /= x.max(dim=0, keepdim=True)[0] + 1e-8
        hue = x[:, 0]
        val = x[:, 1]
        sat = torch.ones_like(hue)  # 固定饱和度
        hsv = torch.stack([hue, sat, val], dim=1)
        rgb = hsv_to_rgb(hsv)
        return rgb

    elif C == 3:
        # === 直接归一化为 RGB ===
        x = features
        x -= x.min(dim=0, keepdim=True)[0]
        x /= x.max(dim=0, keepdim=True)[0] + 1e-8
        return x

    else:
        # === 使用 PCA 降维到 3D ===
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(features.numpy())
        X_3d -= X_3d.min(axis=0, keepdims=True)
        X_3d /= X_3d.max(axis=0, keepdims=True) + 1e-8
        return torch.tensor(X_3d, dtype=torch.float32)
        # === 使用 UMAP 降维到 3D ===
        # reducer = UMAP(n_components=3, metric='cosine')
        # X_3d = reducer.fit_transform(features.numpy())
        # X_3d -= X_3d.min(axis=0, keepdims=True)
        # X_3d /= X_3d.max(axis=0, keepdims=True) + 1e-8
        # return torch.tensor(X_3d, dtype=torch.float32)

def labels_to_color(labels, colormap='tab20'):
    """
    将整数标签映射为颜色 (RGB)。

    Args:
        labels (torch.Tensor): [N]，整数类型的类别标签。
        colormap (str or matplotlib colormap or torch.Tensor): 
            - 字符串，如 'tab10', 'viridis', 'rainbow'；
            - matplotlib colormap 对象；
            - 或自定义 Tensor [K, 3]，K个颜色，值在[0,1]。

    Returns:
        torch.Tensor: [N, 3] RGB 颜色张量，float32，范围 [0, 1]。
    """
    if not torch.is_tensor(labels):
        raise TypeError("labels 必须是 torch.Tensor")
    if labels.dim() != 1:
        raise ValueError("labels 应该是一维 Tensor")

    labels = labels.to(torch.int64).detach().cpu()
    unique_labels = torch.unique(labels)
    num_classes = unique_labels.numel()

    # 创建颜色映射
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap, num_classes)
        color_array = torch.tensor(cmap(range(num_classes))[:, :3], dtype=torch.float32)
    elif isinstance(colormap, torch.Tensor):
        if colormap.shape[1] != 3:
            raise ValueError("自定义 colormap 的 shape 应该为 [K, 3]")
        color_array = colormap.float()
    elif hasattr(colormap, '__call__'):
        # matplotlib colormap 对象
        cmap = colormap
        color_array = torch.tensor(cmap(np.linspace(0, 1, num_classes))[:, :3], dtype=torch.float32)
    else:
        raise ValueError("colormap 类型不支持")

    # 将 labels 映射为颜色
    # 需要把 label 映射到 [0, K-1] 的索引
    label_to_index = {label.item(): i for i, label in enumerate(unique_labels)}
    mapped_indices = torch.tensor([label_to_index[l.item()] for l in labels])
    colors = color_array[mapped_indices]

    return colors

def intensity_to_color(intensities, colormap='viridis', vmin=None, vmax=None):
    """
    将浮点强度值映射为 RGB 颜色。

    Args:
        intensities (torch.Tensor): [N] 浮点数张量。
        colormap (str or matplotlib colormap or torch.Tensor): 
            - str: 'viridis', 'plasma', 'hot', 等；
            - matplotlib colormap 对象；
            - torch.Tensor [K, 3] 自定义颜色映射；
        vmin, vmax (float): 强度归一化范围，默认自动根据数据范围确定。

    Returns:
        torch.Tensor: [N, 3]，RGB 颜色张量，范围 [0, 1]。
    """
    if not torch.is_tensor(intensities):
        raise TypeError("intensities 必须是 torch.Tensor")
    if intensities.dim() != 1:
        raise ValueError("intensities 应该是一维")

    x = intensities.detach().cpu().float().numpy()

    # 归一化
    vmin = x.min() if vmin is None else vmin
    vmax = x.max() if vmax is None else vmax
    x_norm = (x - vmin) / (vmax - vmin + 1e-8)
    x_norm = np.clip(x_norm, 0.0, 1.0)

    # 获取 colormap
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
        rgba = cmap(x_norm)[:, :3]  # 丢弃 alpha
        colors = torch.tensor(rgba, dtype=torch.float32)
    elif isinstance(colormap, torch.Tensor):
        # 自定义颜色渐变，线性插值
        K = colormap.shape[0]
        x_scaled = x_norm * (K - 1)
        x_floor = torch.floor(torch.tensor(x_scaled)).long()
        x_ceil = torch.clamp(x_floor + 1, max=K - 1)
        weight = torch.tensor(x_scaled - x_floor.float()).unsqueeze(1)
        colors = (1 - weight) * colormap[x_floor] + weight * colormap[x_ceil]
    elif hasattr(colormap, '__call__'):
        rgba = colormap(x_norm)[:, :3]
        colors = torch.tensor(rgba, dtype=torch.float32)
    else:
        raise ValueError("colormap 类型不支持")

    return colors