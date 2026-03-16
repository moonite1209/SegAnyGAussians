import torch
import os
import json
import sys
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import GaussianModel
from saga_data import RenderDataset, build_scene_index, move_sample_to_device
from torch.utils.data import DataLoader
from torchvision.utils import save_image


parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--render_path", type=str, required=True)
args = parser.parse_args(sys.argv[1:])
pipe = pp.extract(args)
bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

gs_model = GaussianModel(args.sh_degree)
gs_model.load_ply(args.point_cloud_path)
scene_index = build_scene_index(args)
camera_dataset = RenderDataset(
    scene_index,
    indices=list(range(len(scene_index.specs))),
    resolution=getattr(args, "resolution", 1),
)
camera_loader = DataLoader(camera_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)

with open(args.json_path,'r') as f:
    output = json.load(f)
point_labels = torch.tensor(output['point_labels'], device='cuda')
color_map = torch.rand((1000,3), device='cuda')
precompute_color = color_map[point_labels]
smooth_weights=None

os.makedirs(args.render_path, exist_ok=True)
for sample in tqdm(camera_loader):
    move_sample_to_device(sample, "cuda")
    camera = sample.camera
    render_pkg = render(camera, gs_model, pipe, bg_color, override_color = precompute_color)
    image = render_pkg["render"]
    save_image(image, os.path.join(args.render_path, f'{sample.image_name}.jpg'))
