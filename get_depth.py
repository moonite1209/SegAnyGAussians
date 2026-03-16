import torch
from argparse import ArgumentParser
import os

from arguments import ModelParams, PipelineParams
from scene import FeatureGaussianModel
from saga_data import RenderDataset, build_scene_index, depth_to_camera_points, move_sample_to_device

import gaussian_renderer
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.visualization_utils import scalar_to_color
from torchvision.utils import save_image


if __name__ == '__main__':

    parser = ArgumentParser(description="Get depth for images")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--depth_path', default=None, type=str)
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    scene_index = build_scene_index(dataset)

    instance_feature_dim = getattr(dataset, "instance_feature_dim", getattr(dataset, "feature_dim", 32))
    semantic_feature_dim = getattr(dataset, "semantic_feature_dim", instance_feature_dim)
    feature_gaussians = FeatureGaussianModel(dataset.sh_degree, instance_feature_dim, semantic_feature_dim)
    feature_gaussians.load_ply_from_3dgs(dataset.point_cloud_path)
    feature_gaussians.eval()

    depth_path = args.depth_path
    os.makedirs(depth_path, exist_ok=True)
    render_dataset = RenderDataset(
        scene_index,
        indices=list(range(len(scene_index.specs))),
        resolution=getattr(dataset, "resolution", 1),
    )
    dataloader = DataLoader(render_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)

    background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float, device="cuda")

    for sample in tqdm(dataloader):
        move_sample_to_device(sample, "cuda")
        camera = sample.camera
        rendered_pkg = gaussian_renderer.render_with_depth(camera, feature_gaussians, pipe, background)

        depth = rendered_pkg['depth'].detach().cpu().squeeze()
        H, W = depth.shape

        points_in_3D = depth_to_camera_points(depth, camera)
        image = scalar_to_color(points_in_3D[..., 2].flatten()).reshape(H, W, 3)
        save_image(image.permute(2, 0, 1), os.path.join(depth_path, f"{sample.image_name}.jpg"))
