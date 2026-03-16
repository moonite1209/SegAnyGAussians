from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LoadedCheckpointState:
    epoch: int
    global_step: int
    best_metric: float | None


class FeatureCheckpointManager:
    def __init__(self, paths_cfg, checkpoint_cfg, resolved_config: dict[str, Any]):
        self.paths_cfg = paths_cfg
        self.checkpoint_cfg = checkpoint_cfg
        self.resolved_config = resolved_config
        os.makedirs(self.paths_cfg.checkpoints_dir, exist_ok=True)

    @property
    def last_checkpoint_path(self) -> str:
        return os.path.join(self.paths_cfg.checkpoints_dir, "last.pt")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.paths_cfg.checkpoints_dir, "best.pt")

    def epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.paths_cfg.checkpoints_dir, f"epoch_{epoch:04d}.pt")

    def _build_payload(self, model, epoch: int, global_step: int, best_metric: float | None) -> dict[str, Any]:
        optimizer_state = None
        if model.optimizer is not None:
            optimizer_state = model.optimizer.state_dict()
        return {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "model_state": model.capture(),
            "optimizer_state": optimizer_state,
            "config": self.resolved_config,
        }

    def save(
        self,
        model,
        epoch: int,
        global_step: int,
        best_metric: float | None,
        *,
        save_epoch_checkpoint: bool,
        is_best: bool,
    ) -> None:
        payload = self._build_payload(model, epoch=epoch, global_step=global_step, best_metric=best_metric)
        torch.save(payload, self.last_checkpoint_path)
        if save_epoch_checkpoint and self.checkpoint_cfg.keep_epoch_checkpoints:
            torch.save(payload, self.epoch_checkpoint_path(epoch))
        if is_best and self.checkpoint_cfg.save_best:
            torch.save(payload, self.best_checkpoint_path)

    def load(self, path: str, model, training_cfg, device: torch.device) -> LoadedCheckpointState:
        checkpoint = torch.load(path, map_location=device)
        model.restore_feature_training(checkpoint["model_state"], training_args=training_cfg)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None and model.optimizer is not None:
            model.optimizer.load_state_dict(optimizer_state)
        return LoadedCheckpointState(
            epoch=int(checkpoint.get("epoch", 0)),
            global_step=int(checkpoint.get("global_step", 0)),
            best_metric=checkpoint.get("best_metric"),
        )
