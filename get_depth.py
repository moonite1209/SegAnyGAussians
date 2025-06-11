import torch


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import FeatureScene, Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import os
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from utils.visualization_utils import scalar_to_color
from torchvision.utils import save_image

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':

    parser = ArgumentParser(description="Get depth for images")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--depth_path', default=None, type=str)
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    feature_gaussians = FeatureGaussianModel(dataset.sh_degree)
    feature_gaussians.load_ply_from_3dgs(dataset.point_cloud_path)
    feature_gaussians.eval()
    # feature_scene = FeatureScene(dataset, feature_gaussians)
    # cameras = feature_scene.getTrainCameras()

    depth_path = args.depth_path
    os.makedirs(depth_path, exist_ok=True)

    camera_infos = readColmapCameras(read_extrinsics_binary(os.path.join(args.sparse_path, 'images.bin')), 
                            read_intrinsics_binary(os.path.join(args.sparse_path, 'cameras.bin')), 
                            args.images_path)
    cameras = cameraList_from_camInfos(camera_infos, 1, args)

    background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float, device="cuda")

    for it, camera in tqdm(list(enumerate(cameras))):
        rendered_pkg = gaussian_renderer.render_with_depth(camera, feature_gaussians, pipeline.extract(args), background)

        depth = rendered_pkg['depth'].detach() # pixel-wise
        depth = depth.cpu().squeeze()
        H,W = depth.shape

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(H, W, 3).cpu() # pixel-wise 相机坐标系
        points_in_3D[:,:,-1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = H / 2
        cy = W / 2
        fx = cx / np.tan(cameras[0].FoVy / 2)
        fy = cy / np.tan(cameras[0].FoVx / 2)


        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy
        image = scalar_to_color(points_in_3D[...,2].flatten()).reshape(H,W,3)
        save_image(image.permute(2,0,1), os.path.join(depth_path, f"{camera.image_name}.jpg"))
