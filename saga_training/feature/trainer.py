from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaussian_renderer import render, render_contrastive_feature, render_semantic_feature, render_with_depth
from saga_data import FeatureDataset, build_feature_manifest, move_sample_to_device
from scene import FeatureGaussianModel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

from .checkpointing import FeatureCheckpointManager
from .losses import mask_prototype_info_nce_loss, semantic_mask_alignment_loss
from .reporting import FeatureReporter, ValidationVisualization


RenderFn = Callable[..., dict[str, torch.Tensor]]


@dataclass(frozen=True)
class FeatureTrainerDependencies:
    render_rgb: RenderFn = render
    render_instance: RenderFn = render_contrastive_feature
    render_semantic: RenderFn = render_semantic_feature
    render_depth: RenderFn = render_with_depth


@dataclass(frozen=True)
class FeatureTrainingArtifacts:
    output_dir: str
    feature_point_cloud_path: str
    last_checkpoint_path: str
    best_checkpoint_path: str
    metrics_path: str
    run_metadata_path: str
    epochs_completed: int
    global_step: int
    best_validation_loss: float | None


@dataclass(frozen=True)
class StepMetrics:
    total_loss: float
    instance_loss: float
    semantic_loss: float
    batch_time_ms: float
    valid_instance_masks: int
    valid_semantic_masks: int


class FeatureTrainer:
    def __init__(
        self,
        *,
        model: FeatureGaussianModel,
        train_dataloader_factory: Callable[[], DataLoader],
        val_dataloader_factory: Callable[[], DataLoader],
        reporter: FeatureReporter,
        checkpoints: FeatureCheckpointManager,
        training_cfg: Any,
        device: torch.device,
        pipe_cfg: Any,
        background_rgb: torch.Tensor,
        background_instance: torch.Tensor,
        background_semantic: torch.Tensor,
        dependencies: FeatureTrainerDependencies | None = None,
    ):
        self.model = model
        self.train_dataloader_factory = train_dataloader_factory
        self.val_dataloader_factory = val_dataloader_factory
        self.reporter = reporter
        self.checkpoints = checkpoints
        self.training_cfg = training_cfg
        self.device = device
        self.pipe_cfg = pipe_cfg
        self.background_rgb = background_rgb
        self.background_instance = background_instance
        self.background_semantic = background_semantic
        self.dependencies = dependencies or FeatureTrainerDependencies()
        self.global_step = 0
        self.start_epoch = 1
        self.best_validation_loss = math.inf

    def _measure_step(self, fn: Callable[[], StepMetrics]) -> StepMetrics:
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            metrics = fn()
            end_event.record()
            torch.cuda.synchronize()
            return StepMetrics(
                total_loss=metrics.total_loss,
                instance_loss=metrics.instance_loss,
                semantic_loss=metrics.semantic_loss,
                batch_time_ms=float(start_event.elapsed_time(end_event)),
                valid_instance_masks=metrics.valid_instance_masks,
                valid_semantic_masks=metrics.valid_semantic_masks,
            )

        start_time = time.perf_counter()
        metrics = fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return StepMetrics(
            total_loss=metrics.total_loss,
            instance_loss=metrics.instance_loss,
            semantic_loss=metrics.semantic_loss,
            batch_time_ms=elapsed_ms,
            valid_instance_masks=metrics.valid_instance_masks,
            valid_semantic_masks=metrics.valid_semantic_masks,
        )

    def _render_feature_maps(self, sample):
        depth_map = self.dependencies.render_depth(
            sample.camera,
            self.model,
            self.pipe_cfg,
            self.background_rgb,
        )["depth"].detach()
        instance_render = self.dependencies.render_instance(
            sample.camera,
            self.model,
            self.pipe_cfg,
            self.background_instance,
            depth=depth_map,
        )["render"]
        semantic_render = self.dependencies.render_semantic(
            sample.camera,
            self.model,
            self.pipe_cfg,
            self.background_semantic,
            depth=depth_map,
        )["render"]
        return depth_map, instance_render, semantic_render

    def _train_step(self, sample) -> StepMetrics:
        self.model.optimizer.zero_grad(set_to_none=True)
        move_sample_to_device(sample, self.device.type)

        if sample.masks.shape[0] == 0:
            return StepMetrics(
                total_loss=0.0,
                instance_loss=0.0,
                semantic_loss=0.0,
                batch_time_ms=0.0,
                valid_instance_masks=0,
                valid_semantic_masks=0,
            )

        def _run() -> StepMetrics:
            _, instance_render, semantic_render = self._render_feature_maps(sample)
            instance_result = mask_prototype_info_nce_loss(
                sample.masks,
                instance_render,
                temperature=self.training_cfg.loss.instance_temperature,
            )
            semantic_result = semantic_mask_alignment_loss(
                sample.masks,
                sample.labels,
                sample.label_features,
                semantic_render,
            )
            total_loss = (
                self.training_cfg.loss.instance_weight * instance_result.loss
                + self.training_cfg.loss.semantic_weight * semantic_result.loss
            )
            total_loss.backward()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad(set_to_none=True)
            return StepMetrics(
                total_loss=float(total_loss.detach().item()),
                instance_loss=float(instance_result.loss.detach().item()),
                semantic_loss=float(semantic_result.loss.detach().item()),
                batch_time_ms=0.0,
                valid_instance_masks=instance_result.valid_masks,
                valid_semantic_masks=semantic_result.valid_masks,
            )

        return self._measure_step(_run)

    @staticmethod
    def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        return {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
        }

    def _run_train_epoch(self, epoch: int) -> dict[str, float]:
        dataloader = self.train_dataloader_factory()
        progress = tqdm(
            dataloader,
            desc=f"Train {epoch}/{self.training_cfg.loop.epochs}",
            leave=False,
            disable=self.training_cfg.loop.quiet,
        )
        rows: list[dict[str, float]] = []
        for sample in progress:
            metrics = self._train_step(sample)
            self.global_step += 1
            row = {
                "total_loss": metrics.total_loss,
                "instance_loss": metrics.instance_loss,
                "semantic_loss": metrics.semantic_loss,
                "batch_time_ms": metrics.batch_time_ms,
                "valid_instance_masks": float(metrics.valid_instance_masks),
                "valid_semantic_masks": float(metrics.valid_semantic_masks),
            }
            rows.append(row)
            progress.set_postfix(
                {
                    "loss": f"{metrics.total_loss:.4f}",
                    "inst": f"{metrics.instance_loss:.4f}",
                    "sem": f"{metrics.semantic_loss:.4f}",
                }
            )
            self.reporter.log_train_step(self.global_step, row)
        return self._aggregate_metrics(rows)

    def _run_validation(self, epoch: int) -> dict[str, float]:
        dataloader = self.val_dataloader_factory()
        rows: list[dict[str, float]] = []
        visualizations: list[ValidationVisualization] = []

        with torch.no_grad():
            for sample in dataloader:
                move_sample_to_device(sample, self.device.type)
                depth_map, instance_render, semantic_render = self._render_feature_maps(sample)
                rgb_render = self.dependencies.render_rgb(
                    sample.camera,
                    self.model,
                    self.pipe_cfg,
                    self.background_rgb,
                )["render"]
                instance_result = mask_prototype_info_nce_loss(
                    sample.masks,
                    instance_render,
                    temperature=self.training_cfg.loss.instance_temperature,
                )
                semantic_result = semantic_mask_alignment_loss(
                    sample.masks,
                    sample.labels,
                    sample.label_features,
                    semantic_render,
                )
                total_loss = (
                    self.training_cfg.loss.instance_weight * instance_result.loss
                    + self.training_cfg.loss.semantic_weight * semantic_result.loss
                )
                gt_image = torch.clamp(sample.image, 0.0, 1.0)
                rendered_rgb = torch.clamp(rgb_render, 0.0, 1.0)
                rows.append(
                    {
                        "total_loss": float(total_loss.detach().item()),
                        "instance_loss": float(instance_result.loss.detach().item()),
                        "semantic_loss": float(semantic_result.loss.detach().item()),
                        "l1": float(l1_loss(rendered_rgb, gt_image).detach().item()),
                        "psnr": float(psnr(rendered_rgb, gt_image).mean().detach().item()),
                    }
                )
                if (
                    self.reporter.can_log_validation_images
                    and len(visualizations) < self.training_cfg.validation.max_visualizations
                ):
                    visualizations.append(
                        ValidationVisualization(
                            image_name=sample.image_name,
                            ground_truth=gt_image,
                            rendered_rgb=rendered_rgb,
                            instance_feature_map=instance_render,
                            depth_map=depth_map,
                        )
                    )

        self.reporter.log_validation_visualizations(epoch, visualizations)
        return self._aggregate_metrics(rows)

    def _maybe_resume(self) -> None:
        resume_from = self.training_cfg.checkpoint.resume_from
        if not resume_from:
            return
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        state = self.checkpoints.load(resume_from, self.model, self.training_cfg, self.device)
        self.start_epoch = state.epoch + 1
        self.global_step = state.global_step
        if state.best_metric is not None:
            self.best_validation_loss = float(state.best_metric)

    def run(self) -> FeatureTrainingArtifacts:
        self._maybe_resume()
        completed_epoch = self.start_epoch - 1
        try:
            for epoch in tqdm(
                range(self.start_epoch, self.training_cfg.loop.epochs + 1),
                desc="Feature training",
                disable=self.training_cfg.loop.quiet,
            ):
                train_metrics = self._run_train_epoch(epoch)
                epoch_metrics = {
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                }

                validation_metrics = None
                if (
                    epoch % self.training_cfg.validation.run_every_epochs == 0
                    or epoch == self.training_cfg.loop.epochs
                ):
                    validation_metrics = self._run_validation(epoch)
                    epoch_metrics.update(
                        {
                            f"val_{key}": value
                            for key, value in validation_metrics.items()
                        }
                    )

                is_best = False
                if validation_metrics is not None:
                    candidate = validation_metrics["total_loss"]
                    if candidate < self.best_validation_loss:
                        self.best_validation_loss = candidate
                        is_best = True

                self.reporter.log_epoch(epoch, epoch_metrics)
                self.checkpoints.save(
                    self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=None if math.isinf(self.best_validation_loss) else self.best_validation_loss,
                    save_epoch_checkpoint=(
                        epoch % self.training_cfg.checkpoint.save_every_epochs == 0
                        or epoch == self.training_cfg.loop.epochs
                    ),
                    is_best=is_best,
                )
                completed_epoch = epoch

            self.model.save_ply(self.training_cfg.paths.feature_point_cloud_path)
            best_value = None if math.isinf(self.best_validation_loss) else self.best_validation_loss
            self.reporter.write_run_metadata(
                {
                    "device": self.device.type,
                    "epochs_completed": completed_epoch,
                    "global_step": self.global_step,
                    "best_validation_loss": best_value,
                    "feature_point_cloud_path": self.training_cfg.paths.feature_point_cloud_path,
                    "resumed_from": self.training_cfg.checkpoint.resume_from,
                }
            )
            if self.training_cfg.logging.write_metrics_json:
                self.reporter.write_metrics()
            return FeatureTrainingArtifacts(
                output_dir=self.training_cfg.paths.output_dir,
                feature_point_cloud_path=self.training_cfg.paths.feature_point_cloud_path,
                last_checkpoint_path=self.checkpoints.last_checkpoint_path,
                best_checkpoint_path=self.checkpoints.best_checkpoint_path,
                metrics_path=self.training_cfg.paths.metrics_path,
                run_metadata_path=self.training_cfg.paths.run_metadata_path,
                epochs_completed=completed_epoch,
                global_step=self.global_step,
                best_validation_loss=best_value,
            )
        finally:
            self.reporter.close()


def run_feature_training(app_cfg, dependencies: FeatureTrainerDependencies | None = None) -> FeatureTrainingArtifacts:
    trainer = build_feature_trainer(app_cfg, dependencies=dependencies)
    return trainer.run()


def _resolve_split_ids(manifest, validation_cfg):
    train_ids = list(manifest.train_ids)
    if manifest.test_ids:
        return train_ids, list(manifest.test_ids)

    holdout = min(validation_cfg.fallback_holdout_views, max(1, len(train_ids)))
    val_ids = train_ids[:holdout]
    remaining_train = train_ids[holdout:]
    if remaining_train:
        return remaining_train, val_ids
    return train_ids, val_ids


def _make_dataloader_factory(dataset, dataloader_cfg, *, shuffle: bool) -> Callable[[], DataLoader]:
    def _factory() -> DataLoader:
        kwargs = {
            "batch_size": None,
            "shuffle": shuffle,
            "num_workers": dataloader_cfg.num_workers,
            "pin_memory": dataloader_cfg.pin_memory,
        }
        if dataloader_cfg.num_workers > 0:
            kwargs["persistent_workers"] = dataloader_cfg.persistent_workers
            kwargs["prefetch_factor"] = dataloader_cfg.prefetch_factor
        return DataLoader(dataset, **kwargs)

    return _factory


def build_feature_trainer(
    app_cfg,
    *,
    device: torch.device | None = None,
    dependencies: FeatureTrainerDependencies | None = None,
) -> FeatureTrainer:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(app_cfg.training.paths.output_dir, exist_ok=True)
    os.makedirs(app_cfg.training.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(app_cfg.training.paths.final_dir, exist_ok=True)

    model = FeatureGaussianModel(
        app_cfg.model.sh_degree,
        app_cfg.model.instance_feature_dim,
        app_cfg.model.semantic_feature_dim,
    )
    model.bootstrap_from_scene_ply(app_cfg.training.paths.scene_point_cloud_path, device=device)
    model.training_setup_feature_only(app_cfg.training)

    feature_background_value = 1.0 if app_cfg.model.white_background else 0.0
    background_rgb = torch.tensor(
        [1.0, 1.0, 1.0] if app_cfg.model.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    background_instance = torch.full(
        (app_cfg.model.instance_feature_dim,),
        fill_value=feature_background_value,
        dtype=torch.float32,
        device=device,
    )
    background_semantic = torch.full(
        (app_cfg.model.semantic_feature_dim,),
        fill_value=feature_background_value,
        dtype=torch.float32,
        device=device,
    )

    manifest = build_feature_manifest(
        app_cfg.dataset,
        artifacts_dir=app_cfg.training.paths.artifacts_dir,
    )
    label_features_shape = manifest.require_global_asset("label_features").native_shape
    if (
        label_features_shape is not None
        and len(label_features_shape) == 2
        and label_features_shape[1] != app_cfg.model.semantic_feature_dim
    ):
        raise ValueError(
            "Label feature dimension does not match `model.semantic_feature_dim`: "
            f"{label_features_shape[1]} vs {app_cfg.model.semantic_feature_dim}"
        )
    train_ids, val_ids = _resolve_split_ids(manifest, app_cfg.training.validation)
    train_dataset = FeatureDataset(
        manifest,
        indices=train_ids,
        resolution=app_cfg.dataset.resolution,
    )
    val_dataset = FeatureDataset(
        manifest,
        indices=val_ids,
        resolution=app_cfg.dataset.resolution,
    )

    reporter = FeatureReporter(app_cfg.training.paths, app_cfg.training.logging)
    checkpoints = FeatureCheckpointManager(
        app_cfg.training.paths,
        app_cfg.training.checkpoint,
    )
    return FeatureTrainer(
        model=model,
        train_dataloader_factory=_make_dataloader_factory(train_dataset, app_cfg.training.dataloader, shuffle=True),
        val_dataloader_factory=_make_dataloader_factory(val_dataset, app_cfg.training.dataloader, shuffle=False),
        reporter=reporter,
        checkpoints=checkpoints,
        training_cfg=app_cfg.training,
        device=device,
        pipe_cfg=app_cfg.pipe,
        background_rgb=background_rgb,
        background_instance=background_instance,
        background_semantic=background_semantic,
        dependencies=dependencies,
    )
