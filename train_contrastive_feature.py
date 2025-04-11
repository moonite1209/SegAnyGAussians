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
import torch.nn.functional as F


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

    print("Preparing Quantile Transform...")

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
            sam_masks = viewpoint_cam.original_masks.cuda() # bool[masks, h, w]
            background_mask = (sam_masks.sum(dim=0)==0)
            # sam_masks = torch.cat((sam_masks, background_mask[None]))
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features = render_pkg_feat["render"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2 # regularization term, keep aligned on a ray

        rendered_features = F.normalize(F.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0), dim=0).permute(1,2,0)
        mask_features = torch.zeros((sam_masks.shape[0], rendered_features.shape[-1])).to(rendered_features)
        for i in range(len(mask_features)):
            mask_features[i,...] = F.normalize(rendered_features[sam_masks[i]].mean(dim=0), dim=-1)
        backgroud_feature = F.normalize(rendered_features[background_mask].mean(dim=0), dim=-1)
        mask_features = F.normalize(mask_features, dim=-1)
        intra_loss = []
        for sam_mask, mask_feature in zip(sam_masks, mask_features, strict=True):
            intra_loss.append(((rendered_features[sam_mask]@mask_feature)/2+0.5).mean())
        intra_loss = -torch.log(torch.stack(intra_loss).mean())

        inter_mask_sim = torch.einsum('ac, bc -> ab', torch.cat((mask_features, backgroud_feature[None])), torch.cat((mask_features, backgroud_feature[None])))
        inter_loss = torch.triu(inter_mask_sim/2+0.5, diagonal=1).mean()
        min_val = torch.min(feature_gaussians.get_xyz, dim=0).values
        max_val = torch.max(feature_gaussians.get_xyz, dim=0).values
        new_min = 0.0
        new_max = 1.0
        std_point_xyz = (feature_gaussians.get_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        sample_mask = uniform_sample(feature_gaussians.get_xyz, opt.distance_sample_num)
        sample_xyz = std_point_xyz[sample_mask]
        sample_features = feature_gaussians.get_point_features[sample_mask]
        sample_features = F.normalize(sample_features,dim=-1)
        ptp_xyz_distance = torch.norm(sample_xyz[:,None,:] - sample_xyz[None,:,:], dim=-1) # float[fps,fps]
        ptp_feature_sim = torch.einsum('ac, bc -> ab', sample_features, sample_features) # float[scale,fps,fps]
        distance_loss = (ptp_xyz_distance*torch.clamp(ptp_feature_sim,0)).mean()

        loss = intra_loss \
                + inter_loss \
                + opt.rfn * rendered_feature_norm_reg \
                + opt.distance_weight * distance_loss

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
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
