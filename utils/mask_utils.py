import torch
import numpy as np

def on_boundary(mask: torch.Tensor, margin: int = 1) -> bool:
    """
    判断输入的mask是否在图像边界上。

    参数:
        mask (torch.Tensor): 输入的mask张量，形状应为 [H, W] 或 [C, H, W]。
        margin (int): 边界判定的边缘范围，默认为1（即检查最外层像素）。

    返回:
        bool: 如果mask位于图像边界返回True，否则返回False。
    """
    # 确保mask是二维的，如果是多通道则取第一个通道
    if mask.dim() == 3:
        mask = mask[0]

    h, w = mask.shape

    # 检查四个边界的像素是否有非零值
    top = mask[:margin, :].any()
    bottom = mask[-margin:, :].any()
    left = mask[:, :margin].any()
    right = mask[:, -margin:].any()

    return top or bottom or left or right

def get_mask_map(masks:torch.Tensor):
    # 确保输入符合预期的尺寸
    assert len(masks.shape) == 3, "Masks should be of shape [N, H, W]"
    
    # 确保mask是bool类型
    if masks.dtype != torch.bool:
        masks = masks > 0.5

    device = masks.device
    N, H, W = masks.shape
    
    # 创建随机颜色映射表（确保每种mask有不同颜色）
    colormap = torch.tensor(np.random.randint(0, 256, size=(N, 3)), dtype=torch.uint8, device=device)

    # 初始化空白RGB图像
    rgb_image = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    for i in range(N):
        mask = masks[i]
        color = colormap[i].float() / 255.0  # 归一化到 [0, 1]
        # 将mask对应的颜色赋值给rgb_image
        rgb_image[mask] = color

    return rgb_image