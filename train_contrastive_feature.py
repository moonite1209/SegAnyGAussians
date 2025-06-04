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
from gaussian_renderer import render, render_contrastive_feature, render_with_max_contributor
import sys
from scene import FeatureScene, FeatureGaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from utils.mask_utils import get_mask_map, on_boundary
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
import torch.nn.functional as F
from torch import nn
import pytorch3d.ops


import time

from utils.visualization_utils import feature_map_to_image, features_to_color

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from sklearn.preprocessing import QuantileTransformer

import torch

def uniform_sample(N, n_samples, device = 'cuda:0'):
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

def pickCamera(cameras):
    view_stack = None
    while True:
        if not view_stack:
            view_stack = cameras.copy()
        camera = view_stack.pop(randint(0, len(view_stack)-1))
        yield camera

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from):
    print("RFN weight:", opt.rfn)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    tb_writer = prepare_logger(dataset)

    feature_gaussians = FeatureGaussianModel(dataset.sh_degree, dataset.feature_dim)
    feature_gaussians.load_ply(dataset.point_cloud_path)
    feature_gaussians.training_setup(opt)
    feature_gaussians.eval()
    feature_gaussians._instance_feature.requires_grad = True

    scene = FeatureScene(dataset, feature_gaussians, shuffle=False, sample_rate=1.0)

    background = torch.ones([3], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([3], dtype=torch.float32, device="cuda")
    background_feature = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    iterations = opt.iterations
    if not opt.iterations:
        iterations = min(len(scene.getTrainCameras())*10, 10000)
    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1

    for iteration, viewpoint_cam in zip(range(first_iter, iterations + 1), pickCamera(scene.getTrainCameras())):
        with open(args.progress_path, 'w') as f:
            f.write(str((iteration)*100//iterations))
        torch.cuda.synchronize()
        iter_start.record()

        if viewpoint_cam.original_masks is None or viewpoint_cam.original_masks.shape[0] == 0:
            continue
        with torch.no_grad():
            # N_mask, H, W
            sam_masks = viewpoint_cam.original_masks.cuda() # float[masks, h, w]
            N,H,W = sam_masks.shape
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            background_mask = ~sam_masks.any(dim=0)
            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else torch.clamp(torch.tensor(opt.num_sampled_rays / sam_masks[0].numel()), 0, 1)

            sampled_ray = torch.rand(H,W).cuda()
            sampled_ray = sampled_ray < ray_sample_rate # bool[h, w]

            per_pixel_mask_size = sam_masks * sam_masks[:,sampled_ray].count_nonzero(dim=-1)[:,None,None] # mask fill with size, [masks, h, w]
            background_mask_size = background_mask * background_mask[sampled_ray].count_nonzero(dim=-1)[None,None]
            per_pixel_mean_mask_size = (per_pixel_mask_size.sum(dim = 0) + background_mask_size) / (sam_masks.sum(dim = 0) + background_mask) # float[h, w]
            per_sample_mask_size = per_pixel_mean_mask_size[sampled_ray]
            per_sample_weight = 1 / per_sample_mask_size

            # H W
            # per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None] # mask fill with size, [masks, h, w]

            # per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9) # float[h, w]

            # per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray] # sampled pixel mean mask size, float[sampled pixels]


            # pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1) # float[1, sampled pixels] * float[sampled pixels, 1] -> float[sampled pixels, sampled pixels]
            # ptp_max_size = pixel_to_pixel_mask_size.max()
            # pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            # per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            # per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min() + 1e-9) * 9. + 1. # pixel的平均mask size越大其权重越小, float[sampled pixels, sampled pixels]
            
            sam_masks_sample_mask = sam_masks[:, sampled_ray] # bool[masks, sampled pixels]
            background_sample_mask = background_mask[sampled_ray]

            gt_vec = sam_masks_sample_mask.float() # float[masks, sampled pixels], a pixel belong to a mask
            gt_corrs = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
            gt_corrs[gt_corrs != 0] = 1 # float[sampled pixels, sampled pixels], a pixel in the same mask with another pixel

        render_pkg = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background_feature)
        max_contributor = render_with_max_contributor(viewpoint_cam, feature_gaussians, pipe, background, override_color=torch.zeros(feature_gaussians.get_xyz.shape[0],3,dtype=torch.float,device='cuda'))['max_contributor']
        rendered_features = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        norm_loss = (1-rendered_feature_norm)**2 # regularization term, keep aligned on a ray

        sampled_features = rendered_features[:,sampled_ray].permute((1,0)) # float[sampled scales, C, sampled pixels]
        corr = torch.einsum('ac,bc->ab', sampled_features, sampled_features.detach()) # sampled pixel to sampled pixel similarity, float[sampled pixels, sampled pixels]

        sampled_mask_positive = gt_corrs == 1 # two sampled pixels belong to different mask in any sampled scales, bool[sampled pixels, sampled pixels]
        sampled_mask_positive &= ~(background_sample_mask.float()[:,None]@background_sample_mask.float()[None,:]).bool()
        # sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=1)

        sampled_mask_negative = gt_corrs == 0 # two sampled pixels belong to same mask in any sampled scales, bool[sampled pixels, sampled pixels]
        # sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=1)
        per_mask_weight = torch.ones_like(sampled_mask_negative)*(1/N)*sampled_mask_negative
        
        example_num = sampled_mask_positive.sum()+sampled_mask_negative.sum()
        positive_loss = (- corr[sampled_mask_positive]).mean()
        negative_loss = (torch.relu(corr[sampled_mask_negative])).mean()
        comp_loss = torch.zeros_like(corr)
        comp_loss[sampled_mask_positive] = - 10*corr[sampled_mask_positive]
        comp_loss[sampled_mask_negative] = torch.relu(corr[sampled_mask_negative])
        weighted_comp_loss = (comp_loss * per_sample_weight[None,...] * per_mask_weight).sum(dim=-1).mean()

        distance_loss = torch.tensor(0.,device='cuda')
        # min_val = torch.min(feature_gaussians.get_xyz, dim=0).values
        # max_val = torch.max(feature_gaussians.get_xyz, dim=0).values
        # new_min = 0.0
        # new_max = 1.0
        # std_point_xyz = (feature_gaussians.get_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        # sample_mask = uniform_sample(feature_gaussians.get_xyz.shape[0], opt.distance_sample_num)
        # sample_xyz = std_point_xyz[sample_mask]
        # sample_features = feature_gaussians.get_instance_features[sample_mask]
        # sample_scaled_features = F.normalize(sample_features, dim=-1)
        # ptp_xyz_distance = torch.norm(sample_xyz[:,None,:] - sample_xyz[None,:,:], dim=-1) # float[fps,fps]
        # ptp_feature_sim = torch.einsum('ac, bc -> ab', sample_scaled_features, sample_scaled_features) # float[fps,fps]
        # distance_loss = (ptp_xyz_distance*torch.clamp(ptp_feature_sim,0)).mean()

        outview_loss = torch.tensor(0.,device='cuda')
        if iteration > iterations//2:
            for sam_mask in sam_masks:
                # sam_mask = sam_mask.bool()
                if on_boundary(sam_mask):
                    continue
                mask_feature = F.normalize(rendered_features[:,sam_mask].permute((1,0)).mean(dim=0,keepdim=True),dim=-1)
                invisable_feature = feature_gaussians.get_instance_features[~visibility_filter]
                outview_loss += torch.relu(torch.einsum('ac,bc->ab', mask_feature, invisable_feature.detach())).mean()
        # if iteration > iterations//2:
        #     for sam_mask in sam_masks:
        #         sam_mask = sam_mask.bool()
        #         if on_boundary(sam_mask):
        #             continue
        #         max_contributors = torch.unique(max_contributor[sam_mask])
        #         uniform_sample_mask = uniform_sample(max_contributors.shape[0], 2)
        #         sampled_max_contributors = max_contributors[uniform_sample_mask]
        #         sampled_max_contributor_features = F.normalize(feature_gaussians.get_instance_features[sampled_max_contributors], dim=-1)
        #         invisable_feature = F.normalize(feature_gaussians.get_instance_features[~visibility_filter],dim=-1)
        #         outview_loss += torch.relu(torch.einsum('ac,bc->ab', sampled_max_contributor_features, invisable_feature)).mean()


        if opt.positive_weight == -1:
            opt.positive_weight = 2*(sampled_mask_negative.sum()/example_num).item()
        if opt.negative_weight == -1:
            opt.negative_weight = 2*(sampled_mask_positive.sum()/example_num).item()
        loss = weighted_comp_loss + opt.rfn * norm_loss + opt.distance_weight * distance_loss + outview_loss
        positive_loss = weighted_comp_loss

        with torch.no_grad():
            pos_sim = corr[gt_corrs == 1].mean()
            neg_sim = corr[gt_corrs == 0].mean()

        loss.backward()

        iter_end.record()
        iter_end.synchronize()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "pos loss": f"{positive_loss.item():.{3}f}",
                "neg loss": f"{negative_loss.item():.{3}f}",
                "rfn loss": f"{norm_loss.item():.{3}f}",
                "dis loss": f"{distance_loss.item():.{3}f}",
                "outview loss": f"{outview_loss.item():.{3}f}",
                "loss": f"{loss.item():.{3}f}",
                "pos sim": f"{pos_sim.item():.{3}f}",
                "neg sim": f"{neg_sim.item():.{3}f}",
                "pos weight": f"{opt.positive_weight:.{3}f}",
                "neg weight": f"{opt.negative_weight:.{3}f}",
                "rfn weight": f"{opt.rfn:.{3}f}",
                "dis weight": f"{opt.distance_weight:.{3}f}",
            })
            progress_bar.update(10)

        training_report(tb_writer, testing_iterations, scene, 
                        iteration, loss, positive_loss, negative_loss, norm_loss, distance_loss, outview_loss, iter_start.elapsed_time(iter_end), 
                        get_render_image = lambda viewpoint: render(viewpoint, feature_gaussians, pipe, background)['render'].detach(), 
                        get_feature_map = lambda viewpoint: render_contrastive_feature(viewpoint, feature_gaussians, pipe, background_feature)['render'].detach())

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad()

    feature_gaussians.save_ply(args.contrastive_feature_point_cloud_path)

def prepare_logger(args):    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and os.path.exists(args.log_path):
        tb_writer = SummaryWriter(args.log_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, testing_iterations, scene: FeatureScene, iteration, loss, positive_loss, negative_loss, norm_loss, distance_loss, outview_loss, iter_time, get_render_image, get_feature_map):
    if tb_writer is None:
        return
    if tb_writer:
        tb_writer.add_scalar('train_loss/loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss/positive_loss', positive_loss.item(), iteration)
        tb_writer.add_scalar('train_loss/negative_loss', negative_loss.item(), iteration)
        tb_writer.add_scalar('train_loss/norm_loss', norm_loss.item(), iteration)
        tb_writer.add_scalar('train_loss/distance_loss', distance_loss.item(), iteration)
        tb_writer.add_scalar('train_loss/outview_loss', outview_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', iter_time, iteration)
        tb_writer.add_scalar('grad/norm/mean', scene.feature_gaussians._instance_feature.grad.norm(dim=-1).mean(), iteration)
        tb_writer.add_scalar('grad/norm/std', scene.feature_gaussians._instance_feature.grad.norm(dim=-1).std(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cameras = scene.getTrainCameras()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [train_cameras[idx % len(train_cameras)] for idx in range(0, len(train_cameras), len(train_cameras)//10+1)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = get_render_image(viewpoint)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask_map = get_mask_map(viewpoint.original_masks).permute(2,0,1)
                    feature_map = get_feature_map(viewpoint)
                    C,H,W = feature_map.shape
                    if tb_writer:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/image/render", image[None], global_step=iteration)
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/feature/render", features_to_color(feature_map.reshape(C,-1).permute(1,0)).permute(1,0).reshape(-1,H,W)[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/image/ground_truth", gt_image[None], global_step=iteration)
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/feature/ground_truth", mask_map[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.feature_gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.feature_gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_mesh(f'point_features', scene.feature_gaussians.get_xyz[None], colors=features_to_color(scene.feature_gaussians.get_instance_features)[None]*255, global_step=iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000,2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.images_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
