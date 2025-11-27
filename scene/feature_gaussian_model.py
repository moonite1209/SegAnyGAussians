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

import torch
import pytorch3d.ops
import numpy as np
from scene.dataset_readers import fetchPly
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene import GaussianModel

class FeatureGaussianModel(GaussianModel):

    def setup_functions(self):
        super().setup_functions()
        self.instance_feature_activation = F.normalize
        self.semantic_feature_activation = F.normalize


    def __init__(self, sh_degree: int, instance_feature_dim: int = 32, semantic_feature_dim: int = 32):
        super().__init__(sh_degree)
        self.instance_feature_dim = instance_feature_dim
        self._instance_feature = torch.empty(0)
        self.semantic_feature_dim = semantic_feature_dim
        self._semantic_feature = torch.empty(0)
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
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args = None):
        (self.active_sh_degree, 
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
        self.spatial_lr_scale) = model_args
        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_instance_features(self):
        return self.instance_feature_activation(self._instance_feature)
    
    @property
    def get_semantic_features(self):
        return self.semantic_feature_activation(self._instance_feature)
    
    @property
    def get_std(self):
        return torch.abs(self._std)
    
    def parameters(self):
        return [*super().parameters(), self._instance_feature, self._semantic_feature, self._std]

    def create_from_pcd(self, path, spatial_lr_scale : float):
        pcd = fetchPly(path)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        instance_feature = torch.randn((fused_point_cloud.shape[0], self.instance_feature_dim), dtype=torch.float, device="cuda")
        semantic_feature = torch.randn((fused_point_cloud.shape[0], self.semantic_feature_dim), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        self._instance_feature == nn.Parameter(instance_feature)
        self._semantic_feature == nn.Parameter(semantic_feature)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._instance_feature], 'lr': training_args.instance_feature_lr, "name": "instance_feature"},
            {'params': [self._semantic_feature], 'lr': training_args.semantic_feature_lr, "name": "semantic_feature"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._std], 'lr': training_args.std_lr, "name": "std"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._instance_feature.shape[1]):
            l.append('instance_feature_{}'.format(i))
        for i in range(self._semantic_feature.shape[1]):
            l.append('semantic_feature_{}'.format(i))
        return l

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

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, instance_feature, semantic_feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity_name = [p.name for p in plydata.elements[0].properties if p.name == "opacity"]
        assert len(opacity_name)==1
        opacities = np.zeros((xyz.shape[0], 1))
        for idx, attr_name in enumerate(opacity_name):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])

        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc_")]
        dc_f_names = sorted(dc_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(dc_f_names)==3
        features_dc = np.zeros((xyz.shape[0], len(dc_f_names)))
        for idx, attr_name in enumerate(dc_f_names):
            features_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_dc = features_dc.reshape((features_dc.shape[0], 3, 1))

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        instance_feature_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("instance_feature_")]
        instance_feature_name = sorted(instance_feature_name, key = lambda x: int(x.split('_')[-1]))
        instance_feature = np.random.randn(xyz.shape[0], self.instance_feature_dim)
        for idx, attr_name in enumerate(instance_feature_name):
            instance_feature[:, idx] = np.asarray(plydata.elements[0][attr_name])

        semantic_feature_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("semantic_feature_")]
        semantic_feature_name = sorted(semantic_feature_name, key = lambda x: int(x.split('_')[-1]))
        semantic_feature = np.random.randn(xyz.shape[0], self.semantic_feature_dim)
        for idx, attr_name in enumerate(semantic_feature_name):
            semantic_feature[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))
        self._instance_feature = nn.Parameter(torch.tensor(instance_feature, dtype=torch.float, device="cuda").contiguous())
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").contiguous())
        self._std = nn.Parameter(torch.tensor((0.05), dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree

    def load_ply_from_3dgs(self, path):
        self.load_ply(path)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

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
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_instance_feature, new_semantic_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "instance_feature" : new_instance_feature,
        "semantic_feature" : new_semantic_feature}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._instance_feature = optimizable_tensors["instance_feature"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_instance_feature = self._instance_feature[selected_pts_mask].repeat(N,1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_instance_feature, new_semantic_feature)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_instance_feature = self._instance_feature[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_instance_feature, new_semantic_feature)