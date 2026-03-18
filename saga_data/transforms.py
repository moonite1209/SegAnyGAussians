from __future__ import annotations

import torch
import torch.nn.functional as F


def _target_hw(target_size: tuple[int, int]) -> tuple[int, int]:
    target_w, target_h = target_size
    return int(target_h), int(target_w)


def resize_rgb_tensor(image: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = _target_hw(target_size)
    if tuple(image.shape[-2:]) == (target_h, target_w):
        return image.float()
    resized = F.interpolate(
        image.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp_(0.0, 1.0)


def resize_alpha_tensor(alpha: torch.Tensor | None, target_size: tuple[int, int]) -> torch.Tensor | None:
    if alpha is None:
        return None
    target_h, target_w = _target_hw(target_size)
    if tuple(alpha.shape[-2:]) == (target_h, target_w):
        return alpha.float()
    return F.interpolate(
        alpha.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp_(0.0, 1.0)


def resize_mask_tensor(masks: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = _target_hw(target_size)
    if masks.shape[0] == 0:
        return torch.empty((0, target_h, target_w), dtype=torch.bool)
    if tuple(masks.shape[-2:]) == (target_h, target_w):
        return masks.bool()
    resized = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(target_h, target_w),
        mode="nearest",
    ).squeeze(1)
    return (resized > 0.5).bool()


def resize_depth_tensor(depth: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = _target_hw(target_size)
    if tuple(depth.shape[-2:]) == (target_h, target_w):
        return depth.float()
    return F.interpolate(
        depth.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="nearest",
    ).squeeze(0)
