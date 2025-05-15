import torch

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