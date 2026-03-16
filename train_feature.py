#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from saga_config import TrainingAppConfig
from saga_training.feature import run_feature_training
from utils.general_utils import safe_state


@hydra.main(config_path="configs", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    app_cfg = TrainingAppConfig(**OmegaConf.to_container(cfg, resolve=True))

    print("Optimizing " + app_cfg.dataset.images_path)

    safe_state(app_cfg.training.loop.quiet)
    torch.autograd.set_detect_anomaly(app_cfg.training.loop.detect_anomaly)

    artifacts = run_feature_training(app_cfg)
    print("\nTraining complete.")
    print(f"Feature point cloud: {artifacts.feature_point_cloud_path}")


if __name__ == "__main__":
    main()
