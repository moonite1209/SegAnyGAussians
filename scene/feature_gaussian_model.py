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

import os
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from scene.dataset_readers import fetchPly
from utils.general_utils import get_expon_lr_func
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

from .gaussian_model import GaussianModel


class FeatureGaussianModel(GaussianModel):
    _POINTWISE_GROUPS = {
        "xyz",
        "f_dc",
        "f_rest",
        "opacity",
        "scaling",
        "rotation",
        "instance_feature",
        "semantic_feature",
    }

    def setup_functions(self):
        super().setup_functions()
        self.instance_feature_activation = F.normalize
        self.semantic_feature_activation = F.normalize

    def __init__(self, sh_degree: int, instance_feature_dim: int = 32, semantic_feature_dim: int = 32):
        super().__init__(sh_degree)
        self.instance_feature_dim = instance_feature_dim
        self.semantic_feature_dim = semantic_feature_dim
        self._instance_feature = torch.empty(0)
        self._semantic_feature = torch.empty(0)

    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _model_device(self) -> torch.device:
        if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0:
            return self._xyz.device
        return self._resolve_device()

    def _init_feature_state(self, num_points: int, device: torch.device):
        self._instance_feature = nn.Parameter(
            torch.randn((num_points, self.instance_feature_dim), dtype=torch.float, device=device)
        )
        self._semantic_feature = nn.Parameter(
            torch.randn((num_points, self.semantic_feature_dim), dtype=torch.float, device=device)
        )

    @staticmethod
    def _build_rotation_on_device(rotation: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(
            rotation[:, 0] * rotation[:, 0]
            + rotation[:, 1] * rotation[:, 1]
            + rotation[:, 2] * rotation[:, 2]
            + rotation[:, 3] * rotation[:, 3]
        )
        q = rotation / norm[:, None]

        rot_matrix = torch.zeros((q.size(0), 3, 3), dtype=q.dtype, device=q.device)

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        rot_matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot_matrix[:, 0, 1] = 2 * (x * y - r * z)
        rot_matrix[:, 0, 2] = 2 * (x * z + r * y)
        rot_matrix[:, 1, 0] = 2 * (x * y + r * z)
        rot_matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot_matrix[:, 1, 2] = 2 * (y * z - r * x)
        rot_matrix[:, 2, 0] = 2 * (x * z - r * y)
        rot_matrix[:, 2, 1] = 2 * (y * z + r * x)
        rot_matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot_matrix

    @staticmethod
    def _read_geometry_from_ply(plydata: PlyData, max_sh_degree: int):
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        expected_extra = 3 * (max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) != expected_extra:
            raise ValueError(f"Expected {expected_extra} SH rest attributes, found {len(extra_f_names)}")
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, opacities, features_dc, features_extra, scales, rots

    @staticmethod
    def _read_feature_matrix(plydata: PlyData, prefix: str, expected_dim: int) -> np.ndarray:
        names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
        names = sorted(names, key=lambda x: int(x.split("_")[-1]))
        if len(names) != expected_dim:
            raise ValueError(f"Expected {expected_dim} attributes with prefix `{prefix}`, found {len(names)}")
        feature = np.zeros((plydata.elements[0].count, expected_dim), dtype=np.float32)
        for idx, attr_name in enumerate(names):
            feature[:, idx] = np.asarray(plydata.elements[0][attr_name])
        return feature

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._instance_feature,
            self._semantic_feature,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args=None):
        if len(model_args) == 15:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._instance_feature,
                self._semantic_feature,
                _legacy_std,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        elif len(model_args) == 14:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._instance_feature,
                self._semantic_feature,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        else:
            raise ValueError(f"Unexpected FeatureGaussianModel checkpoint format with {len(model_args)} entries")

        self.optimizer = None
        if training_args is not None:
            self.training_setup(training_args)
            if opt_dict is not None:
                self.optimizer.load_state_dict(opt_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    def restore_feature_training(self, model_args, training_args=None):
        if len(model_args) == 15:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._instance_feature,
                self._semantic_feature,
                _legacy_std,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        elif len(model_args) == 14:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._instance_feature,
                self._semantic_feature,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        else:
            raise ValueError(f"Unexpected FeatureGaussianModel checkpoint format with {len(model_args)} entries")

        self.optimizer = None
        if training_args is not None:
            self.training_setup_feature_only(training_args)
            if opt_dict is not None:
                self.optimizer.load_state_dict(opt_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_instance_features(self):
        return self.instance_feature_activation(self._instance_feature)

    @property
    def get_semantic_features(self):
        return self.semantic_feature_activation(self._semantic_feature)

    def parameters(self):
        return [*super().parameters(), self._instance_feature, self._semantic_feature]

    def set_trainable(self, instance=True, semantic=True, geometry=False):
        for param in super().parameters():
            param.requires_grad = geometry
        self._instance_feature.requires_grad = instance
        self._semantic_feature.requires_grad = semantic

    def create_from_pcd(self, pcd_or_path, spatial_lr_scale: float, device=None):
        if isinstance(pcd_or_path, (str, os.PathLike)):
            pcd = fetchPly(str(pcd_or_path))
        else:
            pcd = pcd_or_path

        device = self._resolve_device(device)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float, device=device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float, device=device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device)), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device=device)
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device)
        )

        self._xyz = nn.Parameter(fused_point_cloud)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        self._init_feature_state(fused_point_cloud.shape[0], device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.active_sh_degree = self.max_sh_degree

    def bootstrap_from_scene_ply(
        self,
        path: str | os.PathLike[str],
        spatial_lr_scale: float | None = None,
        device=None,
    ):
        device = self._resolve_device(device)
        plydata = PlyData.read(path)
        xyz, opacities, features_dc, features_extra, scales, rots = self._read_geometry_from_ply(
            plydata, self.max_sh_degree
        )

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous()
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device))
        self._init_feature_state(xyz.shape[0], device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.active_sh_degree = self.max_sh_degree
        if spatial_lr_scale is not None:
            self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, training_args):
        device = self._model_device()
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        param_groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._instance_feature], "lr": training_args.instance_feature_lr, "name": "instance_feature"},
            {"params": [self._semantic_feature], "lr": training_args.semantic_feature_lr, "name": "semantic_feature"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def training_setup_feature_only(self, training_args):
        device = self._model_device()
        optimizer_cfg = getattr(training_args, "optimizer", training_args)
        self.percent_dense = 0
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        param_groups = [
            {
                "params": [self._instance_feature],
                "lr": optimizer_cfg.instance_feature_lr,
                "name": "instance_feature",
            },
            {
                "params": [self._semantic_feature],
                "lr": optimizer_cfg.semantic_feature_lr,
                "name": "semantic_feature",
            },
        ]

        betas = getattr(optimizer_cfg, "betas", (0.9, 0.999))
        eps = getattr(optimizer_cfg, "eps", 1.0e-15)
        weight_decay = getattr(optimizer_cfg, "weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=0.0,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )
        self.set_trainable(instance=True, semantic=True, geometry=False)

    def construct_list_of_attributes(self):
        attributes = super().construct_list_of_attributes()
        for i in range(self._instance_feature.shape[1]):
            attributes.append(f"instance_feature_{i}")
        for i in range(self._semantic_feature.shape[1]):
            attributes.append(f"semantic_feature_{i}")
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        instance_feature = self._instance_feature.detach().contiguous().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, instance_feature, semantic_feature), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_feature_ply(self, path: str | os.PathLike[str], device=None):
        device = self._resolve_device(device)
        plydata = PlyData.read(path)
        xyz, opacities, features_dc, features_extra, scales, rots = self._read_geometry_from_ply(
            plydata, self.max_sh_degree
        )
        instance_feature = self._read_feature_matrix(plydata, "instance_feature_", self.instance_feature_dim)
        semantic_feature = self._read_feature_matrix(plydata, "semantic_feature_", self.semantic_feature_dim)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous()
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device))
        self._instance_feature = nn.Parameter(torch.tensor(instance_feature, dtype=torch.float, device=device).contiguous())
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device=device).contiguous())
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path, device=None):
        self.load_feature_ply(path, device=device)

    def load_ply_from_3dgs(self, path, spatial_lr_scale=None, device=None):
        self.bootstrap_from_scene_ply(path, spatial_lr_scale=spatial_lr_scale, device=device)

    def _prune_pointwise_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self._POINTWISE_GROUPS:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _cat_pointwise_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self._POINTWISE_GROUPS:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        if self.optimizer is None:
            raise RuntimeError("Cannot prune points before `training_setup()` has created an optimizer")

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_pointwise_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._instance_feature = optimizable_tensors["instance_feature"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_instance_feature,
        new_semantic_feature,
    ):
        if self.optimizer is None:
            raise RuntimeError("Cannot densify points before `training_setup()` has created an optimizer")

        tensors = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "instance_feature": new_instance_feature,
            "semantic_feature": new_semantic_feature,
        }

        optimizable_tensors = self._cat_pointwise_tensors_to_optimizer(tensors)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._instance_feature = optimizable_tensors["instance_feature"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        device = self._model_device()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        device = self._model_device()
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = self._build_rotation_on_device(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_instance_feature = self._instance_feature[selected_pts_mask].repeat(N, 1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_instance_feature,
            new_semantic_feature,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_instance_feature = self._instance_feature[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_instance_feature,
            new_semantic_feature,
        )
