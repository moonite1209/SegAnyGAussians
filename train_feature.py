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
from typing import List
import torch
from random import randint
from gaussian_renderer import render, render_contrastive_feature, render_with_depth, render_with_max_contributor
import sys
from scene import FeatureScene, FeatureGaussianModel
from scene.cameras import Camera
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
from torch.utils.data import DataLoader, random_split
from utils.visualization_utils import feature_map_to_image, features_to_color, scalar_to_color
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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

def groupby_mask(masks, sample_point):
    N,H,W = masks.shape
    ret = []
    for mask in masks:
        ret.append(sample_point[mask[sample_point[:,0], sample_point[:,1]]])
    return ret
def extract_sample(H,W, sample_rate: float, device):
    sample_point = torch.rand(H,W).to(device) < sample_rate
    sample_point = sample_point.nonzero()
    sample_num = len(sample_point)
    return sample_point

def calc_intra_mask_loss(point, rendered_features):
    sample_features = rendered_features[:, point[:,0], point[:,1]]
    sim = torch.einsum('ca, cb -> ab', sample_features, sample_features.detach())
    loss = - sim.mean(dim=1)
    return loss

def calc_inter_mask_loss(point, inter_mask_point, rendered_features):
    features = rendered_features[:, point[:,0], point[:,1]]
    inter_mask_features = rendered_features[:, inter_mask_point[:,0], inter_mask_point[:,1]]
    sim = torch.einsum('ca, cb -> ab', features, inter_mask_features.detach())
    loss = sim.mean(dim=1)
    return loss

def calc_loss(opt, masks, rendered_features):
    N,H,W = masks.shape
    sample_point = extract_sample(H,W, opt.sample_rate, masks.device)
    point_in_mask = filter(lambda p: len(p)>0, groupby_mask(masks, sample_point))
    point_in_background = filter(lambda p: len(p)>0, groupby_mask(~masks.any(dim=0,keepdim=True), sample_point))
    intra_loss = []
    inter_loss = []
    for intra_mask_point in point_in_mask:
        intra_point_loss = calc_intra_mask_loss(intra_mask_point, rendered_features)
        intra_loss.append(intra_point_loss)
        inter_point_loss = []
        for inter_mask_point in [*[p for p in point_in_mask if p is not intra_mask_point], *point_in_background]:
            inter_point_loss.append(calc_inter_mask_loss(intra_mask_point, inter_mask_point, rendered_features))
        inter_point_loss = torch.stack(inter_point_loss, dim=1).mean(dim=1)
        inter_loss.append(inter_point_loss)
    intra_loss = torch.cat(intra_loss, dim=0).mean(dim=0)
    inter_loss = torch.cat(inter_loss, dim=0).mean(dim=0)
    loss = intra_loss + inter_loss
    return loss, intra_loss.detach(), inter_loss.detach()

def batch_report(tb_writer: SummaryWriter, iteration, feature_gaussians, loss, intra_loss, inter_loss, batch_time):
    if tb_writer is None:
        return
    tb_writer.add_scalar('train_loss/loss', loss.item(), iteration)
    tb_writer.add_scalar('train_loss/intra_loss', intra_loss.item(), iteration)
    tb_writer.add_scalar('train_loss/inter_loss', inter_loss.item(), iteration)
    tb_writer.add_scalar('batch_time', batch_time, iteration)

def epoch_report(tb_writer, iteration, val_dataloader, feature_gaussians, get_render_image, get_feature_map, get_depth_map):
    if tb_writer is None:
        return
    l1_test = 0.0
    psnr_test = 0.0
    for idx, cameras in enumerate(val_dataloader):
        for camera in cameras:
            camera.to('cuda')
            image = get_render_image(camera)
            gt_image = torch.clamp(camera.original_image, 0.0, 1.0)
            mask_map = get_mask_map(camera.original_masks).permute(2,0,1)
            depth_map = get_depth_map(camera)
            feature_map = get_feature_map(camera)
            _,H,W = gt_image.shape
            tb_writer.add_images(f"val_view_{camera.image_name}/image/render", image[None], global_step=iteration)
            tb_writer.add_images(f"val_view_{camera.image_name}/feature/render", features_to_color(feature_map.flatten(1).permute(1,0)).permute(1,0).reshape(-1,H,W)[None], global_step=iteration)
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
    psnr_test /= len(val_dataloader) * val_dataloader.batch_size
    l1_test /= len(val_dataloader) * val_dataloader.batch_size
    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, 'val', l1_test, psnr_test))
    tb_writer.add_scalar(f"val/loss_viewpoint - l1_loss", l1_test, iteration)
    tb_writer.add_scalar(f"val/loss_viewpoint - psnr", psnr_test, iteration)
    tb_writer.add_histogram("scene/opacity_histogram", feature_gaussians.get_opacity, iteration)
    tb_writer.add_scalar('total_points', feature_gaussians.get_xyz.shape[0], iteration)
    tb_writer.add_mesh(f'point_features', feature_gaussians.get_xyz[None], colors=features_to_color(feature_gaussians.get_instance_features)[None]*255, global_step=iteration)

def start_report(tb_writer, val_dataloader, feature_gaussians, get_depth_map):
    if tb_writer is None:
        return
    for idx, cameras in enumerate(val_dataloader):
        for camera in cameras:
            camera.to('cuda')
            gt_image = torch.clamp(camera.original_image, 0.0, 1.0)
            mask_map = get_mask_map(camera.original_masks).permute(2,0,1)
            depth_map = get_depth_map(camera)
            _,H,W = gt_image.shape
            tb_writer.add_images(f"val_view_{camera.image_name}/image/ground_truth", gt_image[None], global_step=0)
            tb_writer.add_images(f"val_view_{camera.image_name}/feature/ground_truth", mask_map[None], global_step=0)
            tb_writer.add_images(f"val_view_{camera.image_name}/depth/ground_truth", scalar_to_color(depth_map[0].flatten()).permute(1,0).reshape(-1,H,W)[None], global_step=0)

def train_batch(train_bar, epoch_bar, cameras: List[Camera], scene, feature_gaussians: FeatureGaussianModel, opt, pipe, background, background_feature, tb_writer):
    batch_size = len(cameras)
    batch_timing_start = torch.cuda.Event(enable_timing=True)
    batch_timing_end = torch.cuda.Event(enable_timing=True)

    batch_timing_start.record()
    for i, camera in enumerate(cameras):
        camera.to('cuda')
        N,H,W = camera.original_masks.shape
        render_pkg = render_contrastive_feature(camera, feature_gaussians, pipe, background_feature)
        rendered_features = render_pkg["render"]
        loss, intra_loss, inter_loss = calc_loss(opt, camera.original_masks, rendered_features)
        (loss / batch_size).backward()
    batch_timing_end.record()
    feature_gaussians.optimizer.step()
    epoch_bar.set_postfix({
        "loss": f"{loss.item():.{3}f}",
        "intra_loss": f"{intra_loss.item():.{3}f}",
        "inter_loss": f"{inter_loss.item():.{3}f}",
    })
    batch_report(tb_writer, train_bar.n*epoch_bar.total+epoch_bar.n, feature_gaussians, loss, intra_loss, inter_loss, batch_timing_start.elapsed_time(batch_timing_end))
    feature_gaussians.optimizer.zero_grad()

def train_epoch(train_bar, train_dataloader, val_dataloader, scene, feature_gaussians, opt, pipe, background, background_feature, tb_writer):
    epoch_bar = tqdm(train_dataloader, desc="Epoch progress", position=1, leave=False)
    for cameras in epoch_bar:
        train_batch(train_bar, epoch_bar, cameras, scene, feature_gaussians, opt, pipe, background, background_feature, tb_writer)
    epoch_report(tb_writer, train_bar.n*epoch_bar.total+epoch_bar.n, val_dataloader, feature_gaussians, 
                 get_render_image = lambda viewpoint: render(viewpoint, feature_gaussians, pipe, background)['render'].detach(), 
                 get_feature_map = lambda viewpoint: render_contrastive_feature(viewpoint, feature_gaussians, pipe, background_feature)['render'].detach(), 
                 get_depth_map = lambda viewpoint: render_with_depth(viewpoint, feature_gaussians, pipe, background)['depth'].detach())

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from):

    feature_gaussians = FeatureGaussianModel(dataset.sh_degree, dataset.feature_dim)
    feature_gaussians.load_ply(dataset.point_cloud_path)
    feature_gaussians.training_setup(opt)
    feature_gaussians.eval()
    feature_gaussians._instance_feature.requires_grad_()
    # feature_gaussians._std.requires_grad_()

    scene = FeatureScene(dataset, feature_gaussians, shuffle=False)

    cameras = scene.getCameraDataset()
    num_val = 10
    num_train = len(cameras) - num_val
    train_cameras, val_cameras = random_split(cameras, [num_train, num_val])

    background = torch.tensor((1,1,1) if dataset.white_background else (0,0,0), dtype=torch.float32, device="cuda")
    background_feature = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    # iteration = 0
    # iterations = len(train_cameras)*opt.epochs
    # train_bar = tqdm(range(iteration, iterations), desc="Train progress", position=0)

    tb_writer = prepare_logger(dataset)

    start_report(tb_writer, DataLoader(val_cameras, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=lambda x: x), feature_gaussians, 
                  get_depth_map = lambda camera: render_with_depth(camera, feature_gaussians, pipe, background)['depth'].detach())
    train_bar = tqdm(range(opt.epochs), desc="Train progress", position=0)
    for epoch in train_bar:
        train_dataloader = DataLoader(train_cameras, batch_size=1, shuffle=True, num_workers=os.cpu_count(), collate_fn=lambda x: x)
        val_dataloader = DataLoader(val_cameras, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=lambda x: x)
        train_epoch(train_bar, train_dataloader, val_dataloader, scene, feature_gaussians, opt, pipe, background, background_feature, tb_writer)
    feature_gaussians.save_ply(args.contrastive_feature_point_cloud_path)

    return

def prepare_logger(args):    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and os.path.exists(args.log_path):
        tb_writer = SummaryWriter(args.log_path)
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
