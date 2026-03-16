from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


@dataclass(frozen=True)
class ValidationVisualization:
    image_name: str
    ground_truth: torch.Tensor
    rendered_rgb: torch.Tensor
    instance_feature_map: torch.Tensor | None
    depth_map: torch.Tensor | None


class FeatureReporter:
    def __init__(self, paths_cfg, logging_cfg):
        self.paths_cfg = paths_cfg
        self.logging_cfg = logging_cfg
        self.history: list[dict[str, Any]] = []

        os.makedirs(self.paths_cfg.output_dir, exist_ok=True)
        self.tb_writer = None
        if SummaryWriter is not None and self.logging_cfg.enable_tensorboard:
            os.makedirs(self.paths_cfg.tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(self.paths_cfg.tensorboard_dir)

    @property
    def can_log_validation_images(self) -> bool:
        return self.tb_writer is not None and self.logging_cfg.log_validation_images

    def log_train_step(self, global_step: int, metrics: dict[str, float]) -> None:
        if self.tb_writer is None:
            return
        for name, value in metrics.items():
            self.tb_writer.add_scalar(f"train/{name}", value, global_step)

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        record = {"epoch": epoch, **metrics}
        self.history.append(record)
        if self.tb_writer is not None:
            for name, value in metrics.items():
                if value is None:
                    continue
                self.tb_writer.add_scalar(f"epoch/{name}", value, epoch)
        if self.logging_cfg.write_metrics_json:
            self.write_metrics()

    def log_validation_visualizations(self, epoch: int, visualizations: list[ValidationVisualization]) -> None:
        if not self.can_log_validation_images:
            return

        for item in visualizations:
            gt_image = torch.clamp(item.ground_truth.detach().cpu(), 0.0, 1.0)
            rgb_image = torch.clamp(item.rendered_rgb.detach().cpu(), 0.0, 1.0)
            self.tb_writer.add_images(
                f"val_view_{item.image_name}/image/ground_truth",
                gt_image[None],
                global_step=epoch,
            )
            self.tb_writer.add_images(
                f"val_view_{item.image_name}/image/render",
                rgb_image[None],
                global_step=epoch,
            )
            if item.instance_feature_map is not None:
                feature_map = item.instance_feature_map.detach().cpu()
                _, height, width = feature_map.shape
                try:
                    from utils.visualization_utils import features_to_color

                    feature_colors = features_to_color(feature_map.flatten(1).transpose(0, 1))
                    feature_image = feature_colors.transpose(0, 1).reshape(-1, height, width)
                    self.tb_writer.add_images(
                        f"val_view_{item.image_name}/feature/render",
                        feature_image[None],
                        global_step=epoch,
                    )
                except Exception:
                    pass
            if item.depth_map is not None:
                depth_map = item.depth_map.detach().cpu()
                _, height, width = depth_map.shape
                try:
                    from utils.visualization_utils import scalar_to_color

                    depth_image = scalar_to_color(depth_map[0].flatten()).transpose(0, 1).reshape(-1, height, width)
                    self.tb_writer.add_images(
                        f"val_view_{item.image_name}/depth/render",
                        depth_image[None],
                        global_step=epoch,
                    )
                except Exception:
                    pass

    def write_metrics(self) -> None:
        with open(self.paths_cfg.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        with open(self.paths_cfg.run_metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
