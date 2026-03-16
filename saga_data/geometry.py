from __future__ import annotations

import torch

from .light_camera import LightCamera


def depth_to_camera_points(depth: torch.Tensor, camera: LightCamera) -> torch.Tensor:
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"Expected depth shape [1, H, W], got {tuple(depth.shape)}")
        depth = depth.squeeze(0)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W], got {tuple(depth.shape)}")

    h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = depth.new_tensor(camera.fx)
    fy = depth.new_tensor(camera.fy)
    cx = depth.new_tensor(camera.cx)
    cy = depth.new_tensor(camera.cy)

    points_x = (xs - cx) * depth / fx
    points_y = (ys - cy) * depth / fy
    return torch.stack((points_x, points_y, depth), dim=-1)
