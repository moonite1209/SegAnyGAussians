import shutil
import torch
import os
import json
import sys
from tqdm import tqdm
from gaussian_renderer import render_with_max_contributor
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
import numpy as np
import torch.nn.functional as F
from scene import FeatureScene, FeatureGaussianModel
from utils.general_utils import safe_state
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, pairwise_kernels
from sklearn.preprocessing import minmax_scale, robust_scale, normalize
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax
import hydra
from omegaconf import DictConfig

def uniform_sample(N, n_samples):
    selected_indices = np.random.permutation(N)[:n_samples]
    mask = np.zeros(N, dtype='b1')
    mask[selected_indices] = True
    return mask

def load_model(args, model):
    feature_gaussians = FeatureGaussianModel(model.sh_degree, model.feature_dim)
    feature_gaussians.load_ply(args.feature_point_cloud_path)
    feature_gaussians.eval()
    background_color = torch.tensor([1.]*3 if model.white_background else [0.]*3, dtype=torch.float32, device="cuda")
    background_feature = torch.tensor([0.]*model.feature_dim, dtype=torch.float32, device="cuda")
    return feature_gaussians, background_color, background_feature

def load_cameras(dataset):
    scene = FeatureScene(dataset)
    return scene.getCameraDataset()

def get_sample_mask(total_num, sample_num):
    if sample_num < 0:
        sampled_mask = np.random.rand(total_num) > 0.98
    else:
        sampled_mask = uniform_sample(total_num, sample_num)
    return sampled_mask, sampled_mask.sum().item()

def feature_preprocess(features):
    return features
def xyz_preprocess(xyz):
    return robust_scale(xyz)

def hybird_clustering(args, features, xyzs):
    feature_distance_martix = pairwise_distances(features, metric='cosine')
    xyz_distance_martix = pairwise_distances(xyzs, metric='euclidean')
    distance_martix = args.feature_ratio*feature_distance_martix + (1-args.feature_ratio)*xyz_distance_martix
    clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False, metric='precomputed', n_jobs=-1)
    sample_labels = clusterer.fit_predict(distance_martix)
    clusters = []
    for label in np.unique(sample_labels):
        if label == -3:
            assert False
        elif label == -2:
            assert False
        elif label == -1:
            ...
        else:
            feature_center = normalize(features[sample_labels==label].mean(axis=0, keepdims=True)).squeeze(0)
            xyz_center = xyzs[sample_labels==label].mean(axis=0)
            clusters.append({"label": label, "feature_center": feature_center, "xyz_center": xyz_center})
    return clusters

def distance_to_similarity(distance, gamma):
    return np.exp(-distance*gamma)

def assign_label(args, clusters, features, xyzs):
    P, C = features.shape
    P, D = xyzs.shape
    label = np.array([cluster['label'] for cluster in clusters])
    feature_center = np.array([cluster['feature_center'] for cluster in clusters])
    xyz_center = np.array([cluster['xyz_center'] for cluster in clusters])
    feature_similarity_martix = (pairwise_kernels(features, feature_center, metric='cosine')+1)/2
    xyz_similarity_martix = distance_to_similarity(pairwise_distances(xyzs, xyz_center, metric='euclidean'), 1/D)
    similarity = args.feature_ratio * feature_similarity_martix + (1-args.feature_ratio) * xyz_similarity_martix
    labels = similarity.argmax(axis=1)
    is_valid = similarity.max(axis=1) > args.instance_threshold
    return np.where(is_valid, labels, -1)

def labels_postprocess(args, xyzs, labels):
    knn = KNeighborsClassifier(n_neighbors=args.k, metric='euclidean', n_jobs=-1)
    knn.fit(xyzs, labels)
    return knn.predict(xyzs)

def clustering(args, raw_features: np.ndarray, raw_xyzs: np.ndarray):
    P, C = raw_features.shape
    assert raw_xyzs.shape[0] == P and raw_xyzs.shape[1] == 3
    sample_mask, sample_num = get_sample_mask(P, args.sample_num)
    features = feature_preprocess(raw_features)
    xyzs = xyz_preprocess(raw_xyzs)
    sample_features = features[sample_mask]
    sample_xyzs = xyzs[sample_mask]
    clusters = hybird_clustering(args, sample_features, sample_xyzs)
    labels = assign_label(args, clusters, features, xyzs)
    labels = labels_postprocess(args, xyzs, labels)
    return labels

def choose_class(classes, vote):
    vote, backgound_vote = vote[:-1], vote[-1]
    if vote.max() <= backgound_vote:
        return 'background'
    return classes[vote.argmax().item()]

def assign_class(args, cluster_labels, cameras, feature_gaussians, pipe, background_color, background_feature):
    cluster_to_class = {i.item(): np.zeros(len(args.classes)+1, dtype='i8') for i in np.unique(cluster_labels) if i>=0}
    for camera in tqdm(DataLoader(cameras, batch_size=None, shuffle=False, num_workers=os.cpu_count())):
        masks = camera.original_masks.numpy()
        background_mask = ~masks.any(axis = 0)
        class_labels = camera.labels.numpy()
        camera.to('cuda')
        render_pkg = render_with_max_contributor(camera, feature_gaussians, pipe, background_color)
        max_contributor = render_pkg['max_contributor'].detach().cpu().numpy()
        max_cluster_contributor = cluster_labels[max_contributor]
        for cluster_label in np.unique(max_cluster_contributor):
            if cluster_label < 0:
                continue
            vote = masks[:, max_cluster_contributor==cluster_label].sum(axis=1)
            vote_background = background_mask[max_cluster_contributor==cluster_label].sum(axis=0)
            np.add.at(cluster_to_class[cluster_label], class_labels, vote)
            np.add.at(cluster_to_class[cluster_label], -1, vote_background)
    cluster_to_class = {k: choose_class(args.classes, v) for k, v in cluster_to_class.items()}
    return cluster_to_class

def output_json(args, labels, classes, **kwargs):
    output = {}
    output['point_labels'] = labels
    instances = {str(cluster): {'class': klass} for cluster, klass in classes.items() if klass in args.selected_classes}
    output['instances'] = instances
    for k, v in kwargs:
        output[k] = v
    with open(args.json_path,'w') as f:
        json.dump(output,f)

def clean(args):
    if not args.clean:
        return
    if os.path.isdir(args.masks_path):
        shutil.rmtree(args.masks_path)
    if os.path.isdir(args.labels_path):
        shutil.rmtree(args.labels_path)
    if os.path.isfile(args.feature_point_cloud_path):
        os.remove(args.feature_point_cloud_path)

@hydra.main(config_path="configs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    model = cfg.model
    dataset = cfg.dataset
    pipe = cfg.pipe
    args = cfg.clustering
    
    safe_state(args.quiet)
    feature_gaussians, background_color, background_feature = load_model(args, model)
    cameras = load_cameras(dataset)
    labels = clustering(args, feature_gaussians.get_instance_features.cpu().numpy(), feature_gaussians.get_xyz.cpu().numpy())
    cluster_to_class = assign_class(args, labels, cameras, feature_gaussians, pipe, background_color, background_feature)
    output_json(args, labels.tolist(), cluster_to_class)
    clean(args)

if __name__ == "__main__":
    main()