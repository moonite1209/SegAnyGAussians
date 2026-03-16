import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path

from arguments import ModelParams, PipelineParams
from scene import GaussianModel
from saga_data import RenderDataset, build_scene_index, depth_to_camera_points, move_sample_to_device

import gaussian_renderer
import os
from torch.utils.data import DataLoader
from saga_data.datastore import LocalDataStore
FEATURE_DIM = 32

DATA_ROOT = './data/nerf_llff_data_for_3dgs/'
# MODEL_PATH = './output/figurines_lerf_poses/'
# MODEL_PATH = './output/figurines/'

ALLOW_PRINCIPLE_POINT_SHIFT = False


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
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':

    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    # args = get_combined_args(parser)
    args = parser.parse_args()

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    scene_index = build_scene_index(dataset)

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    scene_gaussians = GaussianModel(dataset.sh_degree)
    scene_gaussians.load_ply(dataset.point_cloud_path)

    assert os.path.exists(dataset.masks_path) and "Please specify a valid masks root."

    OUTPUT_DIR = dataset.mask_scales_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    render_dataset = RenderDataset(
        scene_index,
        indices=list(range(len(scene_index.specs))),
        resolution=getattr(dataset, "resolution", 1),
    )
    dataloader = DataLoader(render_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)
    datastore = LocalDataStore()
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')
    erode_kernel = torch.full((1, 1, 3, 3), 1.0)

    from tqdm import tqdm
    for it, sample in enumerate(tqdm(dataloader, total=len(render_dataset))):
        with open(args.progress_path, 'w') as f:
            f.write(str((it+1)*100//len(render_dataset)))
        move_sample_to_device(sample, "cuda")
        view = sample.camera
        mask_path = Path(dataset.masks_path) / f"{sample.image_name}.pt"
        if not mask_path.exists():
            continue
        rendered_pkg = gaussian_renderer.render_with_depth(view, scene_gaussians, pipe, background)

        depth = rendered_pkg['depth'] # pixel-wise

        corresponding_masks = datastore.load_masks(mask_path, (view.image_width, view.image_height)).cpu()
        if corresponding_masks.numel() == 0:
            continue

        depth = depth.cpu().squeeze()
        points_in_3D = depth_to_camera_points(depth, view).cpu()

        eroded_masks = torch.conv2d(
            corresponding_masks.unsqueeze(1).float(),
            erode_kernel,
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        scale = torch.zeros(len(corresponding_masks))
        for mask_id in range(len(corresponding_masks)):
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
            if point_in_3D_in_mask.numel() == 0:
                continue
            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm() # 对应论文公式(2)

        torch.save(scale, os.path.join(OUTPUT_DIR, sample.image_name + '.pt')) # float[masks]
