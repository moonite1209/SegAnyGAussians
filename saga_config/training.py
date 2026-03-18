from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from .base import StrictConfigModel, require_non_empty

__all__ = [
    "TrainingPathsConfig",
    "TrainingOptimizerConfig",
    "TrainingLoopConfig",
    "TrainingLossConfig",
    "TrainingValidationConfig",
    "TrainingLoggingConfig",
    "TrainingCheckpointConfig",
    "TrainingDataloaderConfig",
    "TrainingConfig",
]


class TrainingPathsConfig(StrictConfigModel):
    scene_point_cloud_path: str
    artifacts_dir: str
    output_dir: str

    @property
    def segment_masks_dir(self) -> str:
        return str(Path(self.artifacts_dir) / "masks")

    @property
    def segment_labels_dir(self) -> str:
        return str(Path(self.artifacts_dir) / "labels")

    @property
    def segment_label_features_path(self) -> str:
        return str(Path(self.segment_labels_dir) / "label_features.pt")

    @property
    def checkpoints_dir(self) -> str:
        return str(Path(self.output_dir) / "checkpoints")

    @property
    def final_dir(self) -> str:
        return str(Path(self.output_dir) / "final")

    @property
    def tensorboard_dir(self) -> str:
        return str(Path(self.output_dir) / "tensorboard")

    @property
    def feature_point_cloud_path(self) -> str:
        return str(Path(self.final_dir) / "feature_point_cloud.ply")

    @property
    def metrics_path(self) -> str:
        return str(Path(self.output_dir) / "metrics.json")

    @property
    def run_metadata_path(self) -> str:
        return str(Path(self.output_dir) / "run_metadata.json")

    @model_validator(mode="after")
    def _validate_semantics(self) -> "TrainingPathsConfig":
        require_non_empty(self.scene_point_cloud_path, "training.paths.scene_point_cloud_path")
        require_non_empty(self.artifacts_dir, "training.paths.artifacts_dir")
        require_non_empty(self.output_dir, "training.paths.output_dir")
        return self


class TrainingOptimizerConfig(StrictConfigModel):
    instance_feature_lr: float = Field(gt=0)
    semantic_feature_lr: float = Field(gt=0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1.0e-15, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _validate_betas(self) -> "TrainingOptimizerConfig":
        beta1, beta2 = self.betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("`training.optimizer.betas` must be in [0, 1)")
        return self


class TrainingLoopConfig(StrictConfigModel):
    epochs: int = Field(gt=0)
    quiet: bool = False
    detect_anomaly: bool = False


class TrainingLossConfig(StrictConfigModel):
    instance_weight: float = Field(default=1.0, ge=0)
    semantic_weight: float = Field(default=1.0, ge=0)
    instance_temperature: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def _validate_weights(self) -> "TrainingLossConfig":
        if self.instance_weight == 0 and self.semantic_weight == 0:
            raise ValueError("At least one training loss weight must be positive")
        return self


class TrainingValidationConfig(StrictConfigModel):
    run_every_epochs: int = Field(default=1, gt=0)
    fallback_holdout_views: int = Field(default=10, gt=0)
    max_visualizations: int = Field(default=4, ge=0)


class TrainingLoggingConfig(StrictConfigModel):
    enable_tensorboard: bool = True
    write_metrics_json: bool = True
    log_validation_images: bool = True


class TrainingCheckpointConfig(StrictConfigModel):
    save_every_epochs: int = Field(default=1, gt=0)
    keep_epoch_checkpoints: bool = True
    save_best: bool = True
    resume_from: str | None = None


class TrainingDataloaderConfig(StrictConfigModel):
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = Field(default=2, gt=0)


class TrainingConfig(StrictConfigModel):
    paths: TrainingPathsConfig
    optimizer: TrainingOptimizerConfig
    loop: TrainingLoopConfig
    loss: TrainingLossConfig
    validation: TrainingValidationConfig
    logging: TrainingLoggingConfig
    checkpoint: TrainingCheckpointConfig
    dataloader: TrainingDataloaderConfig
