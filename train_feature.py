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
from gaussian_renderer import render, render_contrastive_feature, render_with_depth, render_with_max_contributor, render_semantic_feature
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
import hydra
from omegaconf import DictConfig, OmegaConf
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def groupby_mask(masks, sample_point):
    N,H,W = masks.shape
    ret = []
    for mask in masks:
        ret.append(sample_point[mask[sample_point[:,0], sample_point[:,1]]])
    return ret
def extract_sample(H,W, sample_rate: float):
    sample_point = torch.rand(H,W) < sample_rate
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

def calc_loss(args, masks, rendered_features):
    N,H,W = masks.shape
    sample_point = extract_sample(H,W, args.sample_rate).to(masks.device)
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

def calc_loss_mask_avg(args, masks, rendered_features):
    N,H,W = masks.shape
    background_mask = ~masks.any(dim=0)
    background_mask_feature = F.normalize(rendered_features[:, background_mask].mean(dim=1), dim=0)
    mask_features = []
    intra_loss = []
    inter_loss = []
    for mask in masks:
        mask_feature = F.normalize(rendered_features[:, mask].mean(dim=1), dim=0)
        mask_features.append(mask_feature)
        sim = torch.einsum('ca, cb -> ab', rendered_features[:, mask], mask_feature[:,None].detach())
        mask_intra_loss = (-sim).squeeze(1).mean(dim=0)
        intra_loss.append(mask_intra_loss)
    for mask_feature in mask_features:
        other_mask_features = [*[f for f in mask_features if f is not mask_feature], background_mask_feature]
        sim = torch.einsum('ac,bc -> ab', mask_feature[None, ...], torch.stack(other_mask_features, dim=0).detach())
        mask_inter_loss = sim.mean(dim=1).squeeze(0)
        inter_loss.append(mask_inter_loss)
    intra_loss = torch.stack(intra_loss).mean(dim=0)
    inter_loss = torch.stack(inter_loss).mean(dim=0)
    loss = intra_loss + inter_loss
    return loss, intra_loss.detach(), inter_loss.detach()

def calc_InfoNCE_loss(args, masks, rendered_features):
    N,H,W = masks.shape
    background_mask = ~masks.any(dim=0)
    background_mask_feature = F.normalize(rendered_features[:, background_mask].mean(dim=1), dim=0)
    mask_features = []
    for mask in masks:
        mask_feature = F.normalize(rendered_features[:, mask].mean(dim=1), dim=0)
        mask_features.append(mask_feature)
        similarity = torch.einsum('ca, cb -> ab', rendered_features[:, mask], mask_feature[:,None].detach())
    total_intra_item = []
    total_inter_item = []
    for mask, mask_feature in zip(masks, mask_features):
        other_mask_features = [*[f for f in mask_features if f is not mask_feature], background_mask_feature]
        positive_similarity = torch.einsum('ca, cb -> ab', rendered_features[:, mask], mask_feature[:,None].detach())
        intra_item = torch.exp(positive_similarity)
        total_intra_item.append(intra_item.squeeze(1))
        negative_similarity = torch.einsum('ca, cb -> ab', rendered_features[:, mask], torch.stack(other_mask_features, dim=1).detach())
        inter_item = torch.exp(negative_similarity).sum(dim=1, keepdim=True)
        total_inter_item.append(inter_item.squeeze(1))
    total_intra_item = torch.concat(total_intra_item, dim=0)
    total_inter_item = torch.concat(total_inter_item, dim=0)
    loss = -torch.log(total_intra_item/total_inter_item)
    return loss.mean(dim=0), total_intra_item.mean(dim=0).detach(), total_inter_item.mean(dim=0).detach()

def calc_semantic_loss(args, masks, labels, label_features, rendered_features):
    N, H, W = masks.shape
    C, D = label_features.shape  # D 应为 32
    assert D == rendered_features.shape[0], "Feature dimension mismatch"
    assert labels.max() < C and labels.min() >= 0, "Label out of range"

    # 获取每个样本对应的语义特征 [N, 32]
    selected_label_features = label_features[labels]  # shape: [N, 32]

    # 扩展为 [N, 32, H, W] 以与 rendered_features 对齐
    target_features = selected_label_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    # 将 rendered_features 扩展为 [N, 32, H, W]
    rendered_expanded = rendered_features.unsqueeze(0).expand(N, -1, -1, -1)

    # 计算 L1 loss
    l1_loss = torch.abs(rendered_expanded - target_features)  # [N, 32, H, W]

    # 应用 mask：只在 mask 为 True 的位置计算 loss
    mask_expanded = masks.unsqueeze(1).expand_as(l1_loss)  # [N, 32, H, W]
    masked_loss = l1_loss[mask_expanded]

    # 返回平均 loss
    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=l1_loss.device, requires_grad=True)
    return masked_loss.mean()

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
    for idx, camera in enumerate(val_dataloader):
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
    psnr_test /= len(val_dataloader)
    l1_test /= len(val_dataloader)
    tb_writer.add_scalar(f"val/loss_viewpoint - l1_loss", l1_test, iteration)
    tb_writer.add_scalar(f"val/loss_viewpoint - psnr", psnr_test, iteration)
    tb_writer.add_histogram("scene/opacity_histogram", feature_gaussians.get_opacity, iteration)
    tb_writer.add_scalar('total_points', feature_gaussians.get_xyz.shape[0], iteration)
    tb_writer.add_mesh(f'point_features', feature_gaussians.get_xyz[None], colors=features_to_color(feature_gaussians.get_instance_features)[None]*255, global_step=iteration)

def start_report(tb_writer, val_dataloader, feature_gaussians, get_depth_map):
    if tb_writer is None:
        return
    for idx, camera in enumerate(val_dataloader):
        camera.to('cuda')
        gt_image = torch.clamp(camera.original_image, 0.0, 1.0)
        mask_map = get_mask_map(camera.original_masks).permute(2,0,1)
        depth_map = get_depth_map(camera)
        _,H,W = gt_image.shape
        tb_writer.add_images(f"val_view_{camera.image_name}/image/ground_truth", gt_image[None], global_step=0)
        tb_writer.add_images(f"val_view_{camera.image_name}/feature/ground_truth", mask_map[None], global_step=0)
        tb_writer.add_images(f"val_view_{camera.image_name}/depth/ground_truth", scalar_to_color(depth_map[0].flatten()).permute(1,0).reshape(-1,H,W)[None], global_step=0)

def train_batch(train_bar, epoch_bar, camera: Camera, scene, feature_gaussians: FeatureGaussianModel, args, pipe, background, background_feature, tb_writer):
    batch_timing_start = torch.cuda.Event(enable_timing=True)
    batch_timing_end = torch.cuda.Event(enable_timing=True)

    batch_timing_start.record()
    N,H,W = camera.original_masks.shape
    if N==0:
        return
    camera.to('cuda')
    render_pkg = render_contrastive_feature(camera, feature_gaussians, pipe, background_feature)
    rendered_contrastive_features = render_pkg["render"]
    contrastive_loss, intra_loss, inter_loss = calc_loss_mask_avg(args, camera.original_masks, rendered_contrastive_features)
    render_semantic_pkg = render_semantic_feature(camera, feature_gaussians, pipe, background_feature)
    rendered_semantic_features = render_semantic_pkg["render"]
    semantic_loss = calc_semantic_loss(args, camera.original_masks, camera.labels, camera.label_features, rendered_semantic_features)
    loss = contrastive_loss+semantic_loss
    loss.backward()
    batch_timing_end.record()
    feature_gaussians.optimizer.step()
    epoch_bar.set_postfix({
        "loss": f"{loss.item():.{3}f}",
        "intra_loss": f"{intra_loss.item():.{3}f}",
        "inter_loss": f"{inter_loss.item():.{3}f}",
    })
    batch_report(tb_writer, train_bar.n*epoch_bar.total+epoch_bar.n, feature_gaussians, loss, intra_loss, inter_loss, batch_timing_start.elapsed_time(batch_timing_end))
    feature_gaussians.optimizer.zero_grad()

def train_epoch(train_bar, train_dataloader, val_dataloader, scene, feature_gaussians, args, pipe, background, background_feature, tb_writer):
    epoch_bar = tqdm(train_dataloader, desc="Epoch progress", position=1, leave=False)
    for camera in epoch_bar:
        train_batch(train_bar, epoch_bar, camera, scene, feature_gaussians, args, pipe, background, background_feature, tb_writer)
    epoch_report(tb_writer, train_bar.n*epoch_bar.total+epoch_bar.n, val_dataloader, feature_gaussians, 
                 get_render_image = lambda viewpoint: render(viewpoint, feature_gaussians, pipe, background)['render'].detach(), 
                 get_feature_map = lambda viewpoint: render_contrastive_feature(viewpoint, feature_gaussians, pipe, background_feature)['render'].detach(), 
                 get_depth_map = lambda viewpoint: render_with_depth(viewpoint, feature_gaussians, pipe, background)['depth'].detach())

def training(model, dataset, pipe, args):
    feature_gaussians = FeatureGaussianModel(model.sh_degree, model.feature_dim)
    feature_gaussians.load_ply(args.point_cloud_path)
    feature_gaussians.eval()
    feature_gaussians._instance_feature.requires_grad_()
    feature_gaussians._semantic_feature.requires_grad_()
    feature_gaussians.training_setup(args)

    background = torch.tensor([1.]*3 if model.white_background else [0.]*3, dtype=torch.float32, device="cuda")
    background_feature = torch.tensor([0.]*model.feature_dim, dtype=torch.float32, device="cuda")

    scene = FeatureScene(dataset)

    cameras = scene.getTrainDataset()
    num_val = 10
    num_train = len(cameras) - num_val
    train_cameras, val_cameras = random_split(cameras, [num_train, num_val])
    train_cameras = cameras

    tb_writer = prepare_logger(args)

    start_report(tb_writer, DataLoader(val_cameras, batch_size=None, shuffle=False, num_workers=os.cpu_count()), feature_gaussians, 
                  get_depth_map = lambda camera: render_with_depth(camera, feature_gaussians, pipe, background)['depth'].detach())
    train_bar = tqdm(range(args.epochs), desc="Train progress", position=0)
    for epoch in train_bar:
        train_dataloader = DataLoader(train_cameras, batch_size=None, shuffle=True, num_workers=os.cpu_count())
        val_dataloader = DataLoader(val_cameras, batch_size=None, shuffle=False, num_workers=os.cpu_count())
        train_epoch(train_bar, train_dataloader, val_dataloader, scene, feature_gaussians, args, pipe, background, background_feature, tb_writer)
    feature_gaussians.save_ply(args.feature_point_cloud_path)

    return

def prepare_logger(args):    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and os.path.exists(args.tb_path):
        tb_writer = SummaryWriter(args.tb_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@hydra.main(config_path="configs", config_name="training", version_base=None)
def main(cfg : DictConfig):
    model  = cfg.model
    dataset = cfg.dataset
    pipe = cfg.pipe
    args = cfg.training
    
    print("Optimizing " + dataset.images_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model, dataset, pipe, args)

    # All done
    print("\nTraining complete.")

if __name__ == "__main__":
    main()