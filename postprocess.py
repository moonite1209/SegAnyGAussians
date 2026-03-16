import shutil
import torch
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from gaussian_renderer import render_with_max_contributor
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
import numpy as np
import torch.nn.functional as F

from scene import FeatureGaussianModel
from scipy.spatial import KDTree
from hdbscan import HDBSCAN

from utils.general_utils import safe_state
from saga_data import RenderDataset, build_scene_index, move_sample_to_device
from saga_data.datastore import LocalDataStore
from torch.utils.data import DataLoader

def uniform_sample(N, n_samples, device = 'cuda:0'):
    # 生成均匀随机采样的索引
    selected_indices = torch.randperm(N,device=device)[:n_samples]
    # 创建全False的布尔张量
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    # 将选中的位置设置为True
    mask[selected_indices] = True
    return mask

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--progress_path", type=str, required=True)
parser.add_argument("--clean", action='store_true')
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--k", type=int, default=256)
parser.add_argument("--feature_ratio", type=float, default=1.0)
parser.add_argument("--instance_threshold", type=float, default=0.25)
parser.add_argument("--background_threshold", type=float, default=0.5)
parser.add_argument("--scale_threshold", type=float, default=0.8)
parser.add_argument("--opcity_threshold", type=float, default=0.01)
parser.add_argument("--sample_num", type=int, default=-1)
parser.add_argument("--classes", nargs="+", type=str, default=['chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa', 'cabinet', 'bed', 'wall', 'floor', 'ceiling', 'person'])
args = parser.parse_args(sys.argv[1:])
pipe = pp.extract(args)
safe_state(args.quiet)

bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

feat_gs_model = FeatureGaussianModel(args.sh_degree, args.instance_feature_dim, args.semantic_feature_dim)
feat_gs_model.load_ply(args.contrastive_feature_point_cloud_path)
scene_index = build_scene_index(args)
camera_dataset = RenderDataset(
    scene_index,
    indices=list(range(len(scene_index.specs))),
    resolution=getattr(args, "resolution", 1),
)
camera_loader = DataLoader(camera_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)
datastore = LocalDataStore()

point_features = feat_gs_model.get_instance_features.detach().cpu()
point_xyz = feat_gs_model.get_xyz.detach().cpu()
point_scales = feat_gs_model.get_scaling.detach().cpu()
is_big_gaussian = point_scales.max(dim=-1).values>point_scales.max(dim=-1).values.median()*args.scale_threshold
point_opacities = feat_gs_model.get_opacity.detach().cpu().squeeze()
is_transparent_gaussian = point_opacities<args.opcity_threshold
print(f'{point_features.shape=}, {point_xyz.shape=}')

sample_num = args.sample_num
if sample_num < 0:
    sampled_mask = torch.rand(point_xyz.shape[0]) > 0.98
    sample_num = sampled_mask.sum()
else:
    sampled_mask = uniform_sample(point_xyz.shape[0], sample_num, device='cpu')

sampled_point_features = point_features[sampled_mask]

min_val = torch.min(point_xyz, dim=0).values
max_val = torch.max(point_xyz, dim=0).values
new_min = 0.0
new_max = 1.0
std_point_xyz = (point_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
sampled_std_point_xyz = std_point_xyz[sampled_mask]

hybird_point_features = torch.cat((point_features, std_point_xyz), dim=1)
sampled_hybird_point_features = torch.cat((sampled_point_features, sampled_std_point_xyz), dim=1)

def get_hybird_sim(a, b):
    point_feature1 = a[:-3]
    std_point_xyz1 = a[-3:]
    point_feature2 = b[:-3]
    std_point_xyz2 = b[-3:]
    feature_sim = np.clip((np.dot(point_feature1, point_feature2)+1)/2, 0, 1)
    std_xyz_sim = np.clip(np.exp(-np.linalg.norm(std_point_xyz1 - std_point_xyz2)), 0, 1)
    return args.feature_ratio * feature_sim + (1-args.feature_ratio) * std_xyz_sim
clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.00000001, allow_single_cluster = False, metric='precomputed', core_dist_n_jobs=-1) # HDBSCAN

sampled_point_features_distance = torch.clamp(1-torch.einsum('ac,bc -> ab', sampled_point_features, sampled_point_features), 0)
sampled_std_point_xyz_distance = torch.clamp(torch.norm(sampled_std_point_xyz[:,None,:] - sampled_std_point_xyz[None,:,:], dim=-1), 0)
sampled_hybird_distance = args.feature_ratio*sampled_point_features_distance + (1-args.feature_ratio)*sampled_std_point_xyz_distance
cluster_labels = clusterer.fit_predict(sampled_hybird_distance.numpy().astype(np.float64))

feature_cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, point_features.shape[-1])
xyz_cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, point_xyz.shape[-1])
for i in np.unique(cluster_labels):
    if i<0:
        continue
    feature_cluster_centers[i] = F.normalize(sampled_point_features[cluster_labels == i].mean(dim = 0),dim=-1)
    xyz_cluster_centers[i] = sampled_std_point_xyz[cluster_labels == i].mean(dim = 0)

point_features_sim = torch.clamp(torch.einsum('ac,bc->ab', point_features, feature_cluster_centers)/2+0.5, 0, 1)
std_point_xyz_sim = torch.clamp(torch.exp(-torch.norm(std_point_xyz[:,None,:] - xyz_cluster_centers[None,:,:], dim=-1)), 0, 1)
hybird_sim = args.feature_ratio*point_features_sim + (1-args.feature_ratio)*std_point_xyz_sim
confidence = torch.softmax(hybird_sim*10, dim=-1)
mask, point_labels = confidence.max(dim=-1)
mask = mask>args.instance_threshold
print(f'{mask.sum()=}, {(~mask).sum()=}')
point_labels[~mask] = -1
print(f'HDBSCAN finish')
def filter3d(pos, label, k):
    print('begin filter3d')
    assert pos.shape[0] == label.shape[0]
    pos=pos.detach().cpu().numpy()
    label=label.detach().cpu().numpy()
    new_label = []
    kdtree = KDTree(pos)
    for i,p in enumerate(pos):
        d, index = kdtree.query(x=p, k=k)
        # assert i == index[0]
        # print(f'query index {index[1:]} for {index[0]}')
        # print(f'query label {label[index[1:]].tolist()} for {label[index[0]].tolist()}')
        # index = index[1:]
        bin = []
        counts = []
        for l in label[index]:
            try:
                counts[bin.index(l)]+=1
            except:
                bin.append(l)
                counts.append(1)
        # print(f'{bin}\n{counts}')
        new_label.append(bin[counts.index(max(counts))])
    print('finish filter3d')
    return torch.tensor(new_label)
if args.k>0:
    point_labels = filter3d(point_xyz, point_labels, args.k)
end_time = datetime.now()
print(f'knn finish')

point_labels = point_labels.numpy()
instances = np.unique(point_labels).tolist()
vote = {instance: [0 for _ in range(len(args.classes)+1)] for instance in instances}
contribute = torch.zeros((point_xyz.shape[0]), dtype=torch.float32, requires_grad=False)
for i, sample in enumerate(tqdm(camera_loader, total=len(camera_dataset))):
    with open(args.progress_path, 'w') as f:
        f.write(str((i+1)*100//len(camera_dataset)))
    mask_path = os.path.join(args.masks_path, f'{sample.image_name}.pt')
    label_path = os.path.join(args.labels_path, f'{sample.image_name}.pt')
    if not os.path.exists(mask_path) or not os.path.exists(label_path):
        continue
    move_sample_to_device(sample, "cuda")
    camera = sample.camera
    masks = datastore.load_masks(Path(mask_path), (camera.image_width, camera.image_height)).numpy()
    if masks.shape[0] == 0:
        continue
    labels = torch.load(label_path, weights_only=True).numpy()
    render_pkg = render_with_max_contributor(camera, feat_gs_model, pipe, bg_color)
    max_contributor = render_pkg['max_contributor'].detach().cpu().numpy()
    contribute += render_pkg['contribute'].detach().cpu()
    max_instance_contributor = point_labels[max_contributor]
    background_label = len(args.classes)
    background = np.ones_like(masks[0], dtype=bool)
    for label, mask in zip(labels, masks):
        background &= ~mask
        vote_for_label = max_instance_contributor[mask]
        for instance in instances:
            vote[instance][label] += int((vote_for_label == instance).sum())
    vote_for_background_label = max_instance_contributor[background]
    for instance in instances:
        vote[instance][background_label] += int((vote_for_background_label == instance).sum())

def get_class(classes, votes):
    votes = np.array(votes)
    ratio = votes/votes.sum()
    if ratio[-1] == ratio.max() and ratio[-1]>args.background_threshold:
        return 'background'
    return classes[ratio[:-1].argmax()]

output = dict()
output['point_labels'] = point_labels.tolist()
output['is_big_gaussian'] = is_big_gaussian.tolist()
output['is_transparent_gaussian'] = is_transparent_gaussian.tolist()
output['contribute'] = contribute.tolist()
output['instances'] = {instance: {'class': get_class(args.classes, votes)} for instance, votes in vote.items()}
output['instances'] = {k: v for k, v in output['instances'].items() if v.get('class') in ['chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa', 'cabinet', 'bed']}
with open(args.json_path,'w') as f:
    json.dump(output,f)
if(args.clean):
    if os.path.isdir(args.masks_path):
        shutil.rmtree(args.masks_path)
    if os.path.isdir(args.labels_path):
        shutil.rmtree(args.labels_path)
    if os.path.isfile(args.contrastive_feature_point_cloud_path):
        os.remove(args.contrastive_feature_point_cloud_path)
