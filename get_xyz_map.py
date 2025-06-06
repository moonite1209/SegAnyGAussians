import torch


import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureScene, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

import os
from tqdm import tqdm
FEATURE_DIM = 32

def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
        # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
        # print(k, merged_dict[k])
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid(torch.arange(h), torch.arange(w),indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':

    parser = ArgumentParser(description="Get scales for SAM masks")

    # model = ModelParams(parser, sentinel=True)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    feature_gaussians = GaussianModel(dataset.sh_degree)

    feature_scene = FeatureScene(dataset, feature_gaussians)


    xyz_map_path = dataset.xyz_map_path
    os.makedirs(xyz_map_path, exist_ok=True)

    cameras = feature_scene.getTrainCameras()

    for it, view in tqdm(list(enumerate(cameras))):
        with open(args.progress_path, 'w') as f:
            f.write(str((it+1)*100//len(cameras)))
        rendered_pkg = gaussian_renderer.render_with_depth(view, feature_gaussians, pipe, background)

        depth = rendered_pkg['depth'].detach() # pixel-wise

        if view.original_masks == None:
            continue
        masks = view.original_masks

        depth = depth.cpu().squeeze()

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu() # pixel-wise 相机坐标系
        points_in_3D[:,:,-1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(cameras[0].FoVx / 2)
        fy = cy / np.tan(cameras[0].FoVy / 2)


        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

        # upsampled_mask = torch.nn.functional.interpolate(masks.unsqueeze(1), mode = 'bilinear', size = (depth.shape[0], depth.shape[1]), align_corners = False)
        # eroded_masks = torch.conv2d(
        #     upsampled_mask.float(),
        #     torch.full((3, 3), 1.0).view(1, 1, 3, 3).to(upsampled_mask.device),
        #     padding=1,
        # )
        # eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        scale = torch.zeros(len(masks))
        for mask_id in range(len(masks)):
            
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]

            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm() # 对应论文公式(2)

        torch.save(scale, os.path.join(xyz_map_path, view.image_name + '.pt')) # float[masks]