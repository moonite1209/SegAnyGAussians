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
import torch
from random import randint
from gaussian_renderer import render_contrastive_feature
import sys
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
from torch import nn
import pytorch3d.ops


import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from sklearn.preprocessing import QuantileTransformer

import torch

def uniform_sample(xyz, n_samples):
    device = xyz.device
    N, _ = xyz.shape
    # 生成均匀随机采样的索引
    selected_indices = torch.randperm(N,device=device)[:n_samples]
    # 创建全False的布尔张量
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    # 将选中的位置设置为True
    mask[selected_indices] = True
    return mask

def farthest_point_sample(xyz, n_samples):
    """
    输入：
        xyz:       点云坐标，形状为 [N, 3] 的PyTorch张量
        n_samples: 需要采样的点数
    输出：
        mask:      布尔掩码，形状为 [N]，True表示被采样的点
    """
    xyz = xyz.detach().cpu()
    device = xyz.device
    N, _ = xyz.shape
    
    # 初始化采样点索引和距离矩阵
    centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
    distance = torch.full((N,), float('inf'), device=device)
    
    # 随机选择第一个点（或按质心优化选择）
    farthest = torch.randint(0, N, (1,), device=device).item()
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        
        # 计算所有点到当前采样点的欧氏距离
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        
        # 更新每个点的最小距离（与已选点集的最近距离）
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # 选择距离最大的点作为下一个采样点
        farthest = torch.argmax(distance)
    
    # 生成布尔掩码
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[centroids] = True
    return mask

# Borrowed from GARField but modified
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        # This function acts as a wrapper for QuantileTransformer.
        # QuantileTransformer expects a numpy array, while we have a torch tensor.
        scales = scales.reshape(-1,1)
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device)

    return quantile_transformer_func

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_features = False
    dataset.need_masks = True
    dataset.allow_principle_point_shift = False

    gaussians = None #GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    if not opt.iterations:
        opt.iterations = min(len(scene.getTrainCameras())*10, 10000)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        with open(args.progress_path, 'w') as f:
            f.write(str(50+(iteration)*25//opt.iterations))
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if iteration < -1:
            viewpoint_cam = viewpoint_stack[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        while viewpoint_cam.original_masks==None:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            
            if iteration < -1:
                viewpoint_cam = viewpoint_stack[0]
            else:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        with torch.no_grad():
            # N_mask, H, W
            sam_masks = viewpoint_cam.original_masks.cuda().float() # float[masks, h, w]
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            background_mask = (sam_masks.sum(dim = 0) == 0).float()
            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else torch.clamp(torch.tensor(opt.num_sampled_rays / (sam_masks.shape[-2]*sam_masks.shape[-1])), 0, 1)

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda()
            sampled_ray = sampled_ray < ray_sample_rate
            # H W
            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None] # mask fill with size, [masks, h, w]

            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9) # float[h, w]

            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray] # sampled pixel mean mask size, float[sampled pixels]


            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1) # float[1, sampled pixels] * float[sampled pixels, 1] -> float[sampled pixels, sampled pixels]
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min() + 1e-9) * 9. + 1. # pixel的平均mask size越大其权重越小, float[sampled pixels, sampled pixels]
            
            sam_masks_sampled_ray = sam_masks[:, sampled_ray] # bool[masks, sampled pixels]
            background_sample_mask = background_mask[sampled_ray]

            gt_vec = sam_masks_sampled_ray # float[masks, sampled pixels], a pixel belong to a mask
            gt_corrs = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
            gt_corrs[gt_corrs != 0] = 1 # float[sampled pixels, sampled pixels], a pixel in the same mask with another pixel

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features = render_pkg_feat["render"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2 # regularization term, keep aligned on a ray

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0)

        sampled_feature_with_scale = rendered_features[:,sampled_ray] # float[sampled scales, C, sampled pixels]

        scale_conditioned_features_sam = sampled_feature_with_scale.permute([1,0]) # float[sampled pixels, C]

        scale_conditioned_features_sam = torch.nn.functional.normalize(scale_conditioned_features_sam, dim=-1, p=2)
        corr = torch.einsum('ac,bc->ab', scale_conditioned_features_sam, scale_conditioned_features_sam) # sampled pixel to sampled pixel similarity, float[sampled pixels, sampled pixels]

        sampled_mask_positive = gt_corrs == 1 # two sampled pixels belong to different mask in any sampled scales, bool[sampled pixels, sampled pixels]
        sampled_mask_positive &= ~(background_sample_mask[:,None]@background_sample_mask[None,:]).bool()
        sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=1)

        sampled_mask_negative = gt_corrs == 0 # two sampled pixels belong to same mask in any sampled scales, bool[sampled pixels, sampled pixels]
        sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=1)
        
        example_num = sampled_mask_positive.sum()+sampled_mask_negative.sum()
        positive_loss = (- per_pixel_weight[sampled_mask_positive] * gt_corrs[sampled_mask_positive] * corr[sampled_mask_positive]).mean()
        negative_loss = (per_pixel_weight[sampled_mask_negative] * (1 - gt_corrs[sampled_mask_negative]) * torch.relu(corr[sampled_mask_negative])).mean()
        # inconsistent = torch.logical_not(torch.logical_or(consistent_negative, consistent_positive))
        # inconsistent_num = inconsistent.count_nonzero()
        # sampled_num = inconsistent_num / 2

        # rand_num = torch.rand_like(gt_corrs)

        # sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / consistent_positive.count_nonzero())

        # sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / consistent_negative.count_nonzero())

        # sampled_mask_positive = torch.logical_or(
        #     torch.logical_or(
        #         sampled_positive, torch.any(torch.logical_and(corr < 0.75, gt_corrs == 1), dim = 0)
        #     ), 
        #     inconsistent
        # )
        # sampled_mask_positive = torch.logical_and(sampled_mask_positive, ~diag_mask)
        # sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)
        # sampled_mask_positive = sampled_mask_positive.bool()

        # sampled_mask_negative = torch.logical_or(
        #     torch.logical_or(
        #         sampled_negative, torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0), dim = 0)
        #     ), 
        #     inconsistent
        # )
        # sampled_mask_negative = torch.logical_and(sampled_mask_negative, ~diag_mask)
        # sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
        # sampled_mask_negative = sampled_mask_negative.bool()

        # per_pixel_weight = per_pixel_weight.unsqueeze(0)

        min_val = torch.min(feature_gaussians.get_xyz, dim=0).values
        max_val = torch.max(feature_gaussians.get_xyz, dim=0).values
        new_min = 0.0
        new_max = 1.0
        std_point_xyz = (feature_gaussians.get_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        sample_mask = uniform_sample(feature_gaussians.get_xyz, opt.distance_sample_num)
        sample_xyz = std_point_xyz[sample_mask]
        sample_features = feature_gaussians.get_point_features[sample_mask]
        sample_scaled_features = torch.nn.functional.normalize(sample_features, dim=-1)
        ptp_xyz_distance = torch.norm(sample_xyz[:,None,:] - sample_xyz[None,:,:], dim=-1) # float[fps,fps]
        ptp_feature_sim = torch.einsum('ac, bc -> ab', sample_scaled_features, sample_scaled_features) # float[fps,fps]
        distance_loss = (ptp_xyz_distance*torch.clamp(ptp_feature_sim,0)).mean()

        positive_weight = 2*(sampled_mask_negative.sum()/example_num)
        negative_weight = 2*(sampled_mask_positive.sum()/example_num)
        loss = positive_weight*positive_loss + negative_weight*negative_loss + opt.rfn * rendered_feature_norm_reg + opt.distance_weight * distance_loss

        with torch.no_grad():
            pos_sim = corr[gt_corrs == 1].mean()
            neg_sim = corr[gt_corrs == 0].mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "pos loss": f"{positive_loss.item():.{3}f}",
                "neg loss": f"{negative_loss.item():.{3}f}",
                "rfn loss": f"{rendered_feature_norm_reg.item():.{3}f}",
                "dis loss": f"{distance_loss.item():.{3}f}",
                "loss": f"{loss.item():.{3}f}",
                "pos sim": f"{pos_sim.item():.{3}f}",
                "neg sim": f"{neg_sim.item():.{3}f}",
                "pos weight": f"{2*(sampled_mask_negative.sum()/example_num).item():.{3}f}",
                "neg weight": f"{2*(sampled_mask_positive.sum()/example_num).item():.{3}f}",
                "rfn weight": f"{opt.rfn:.{3}f}",
                "dis weight": f"{opt.distance_weight:.{3}f}",
            })
            progress_bar.update(10)

    
    # scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = opt.smooth_K)
    scene.feature_gaussians.save_ply(args.contrastive_feature_point_cloud_path, 
                                     smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None,
                                     smooth_type = 'traditional', 
                                     smooth_K = opt.smooth_K)
    # torch.save(scale_gate.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}/".format(iteration) + "scale_gate.pt"))
    # torch.save(scale_gate.state_dict(), args.scale_gate_path)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    
    # args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.save_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
