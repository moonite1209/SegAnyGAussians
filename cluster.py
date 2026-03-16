import shutil
import torch
import os
import json
import sys
from tqdm import tqdm
from gaussian_renderer import render_with_max_contributor
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
from scene import FeatureGaussianModel
from utils.general_utils import safe_state
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, pairwise_kernels
from sklearn.preprocessing import minmax_scale, robust_scale, normalize
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from saga_config import ClusteringAppConfig, ClusteringConfig, ModelConfig
import open3d as o3d

def uniform_sample(N, n_samples):
    selected_indices = np.random.permutation(N)[:n_samples]
    mask = np.zeros(N, dtype='b1')
    mask[selected_indices] = True
    return mask

def load_model(args: ClusteringConfig, model: ModelConfig):
    feature_gaussians = FeatureGaussianModel(model.sh_degree, model.instance_feature_dim, model.semantic_feature_dim)
    feature_gaussians.load_ply(args.feature_point_cloud_path)
    feature_gaussians.eval()
    return feature_gaussians

def get_sample_mask(total_num, sample_num):
    if sample_num < 0:
        sampled_mask = np.random.rand(total_num) > 0.97
    else:
        sampled_mask = uniform_sample(total_num, sample_num)
    return sampled_mask, sampled_mask.sum().item()

def feature_preprocess(features):
    return features
def xyz_preprocess(xyz):
    return robust_scale(xyz)

def hybird_clustering(args: ClusteringConfig, instance_features, semantic_features, xyzs):
    
    # Compute distance matrices for instance and semantic features separately
    instance_distance_matrix = pairwise_distances(instance_features, metric='cosine')
    semantic_distance_matrix = pairwise_distances(semantic_features, metric='cosine')
    
    # Get feature ratios from args
    
    xyz_distance_matrix = pairwise_distances(xyzs, metric='euclidean')
    distance_matrix = args.instance_feature_ratio * instance_distance_matrix + args.semantic_feature_ratio * semantic_distance_matrix + args.xyz_feature_ratio * xyz_distance_matrix
    
    clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster=False, metric='precomputed', n_jobs=-1)
    sample_labels = clusterer.fit_predict(distance_matrix)
    
    clusters = []
    for label in np.unique(sample_labels):
        if label == -3:
            assert False
        elif label == -2:
            assert False
        elif label == -1:
            continue
        else:
            # Compute centers for instance and semantic features separately
            instance_center = normalize(instance_features[sample_labels == label].mean(axis=0, keepdims=True)).squeeze(0)
            semantic_center = normalize(semantic_features[sample_labels == label].mean(axis=0, keepdims=True)).squeeze(0)
            xyz_center = xyzs[sample_labels == label].mean(axis=0)
            
            # Store centers separately
            clusters.append({"label": label, "instance_feature_center": instance_center, "semantic_feature_center": semantic_center, "xyz_center": xyz_center})
    return clusters

def distance_to_similarity(distance, gamma):
    return np.exp(-distance*gamma)

def assign_label(args: ClusteringConfig, clusters, instance_features, semantic_features, xyzs):
    _, D = xyzs.shape
    
    # Extract separate feature centers
    instance_center = np.array([cluster['instance_feature_center'] for cluster in clusters])
    semantic_center = np.array([cluster['semantic_feature_center'] for cluster in clusters])
    xyz_center = np.array([cluster['xyz_center'] for cluster in clusters])
    
    # Compute similarity matrices for instance and semantic features separately
    instance_similarity_matrix = (pairwise_kernels(instance_features, instance_center, metric='cosine') + 1) / 2
    semantic_similarity_matrix = (pairwise_kernels(semantic_features, semantic_center, metric='cosine') + 1) / 2
    
    xyz_similarity_matrix = distance_to_similarity(pairwise_distances(xyzs, xyz_center, metric='euclidean'), 1/D)
    
    # Get feature ratios from args
    similarity = args.instance_feature_ratio * instance_similarity_matrix + args.semantic_feature_ratio * semantic_similarity_matrix + args.xyz_feature_ratio * xyz_similarity_matrix
    
    labels = similarity.argmax(axis=1)
    is_valid = similarity.max(axis=1) > args.instance_threshold
    return np.where(is_valid, labels, -1)

def labels_postprocess(args: ClusteringConfig, xyzs, labels):
    knn = KNeighborsClassifier(n_neighbors=args.k, metric='euclidean', n_jobs=-1)
    knn.fit(xyzs, labels)
    return knn.predict(xyzs)

def sor_filter_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """
    Statistical Outlier Removal (SOR) filter to remove outliers from point cloud
    
    Args:
        points: numpy array of shape (N, 3) representing point coordinates
        nb_neighbors: number of neighbors to consider for distance calculation
        std_ratio: standard deviation ratio threshold
    
    Returns:
        inlier_mask: boolean mask indicating which points are inliers
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[ind] = True
    
    return inlier_mask

def clustering(args: ClusteringConfig, raw_instance_features: np.ndarray, raw_semantic_features: np.ndarray, raw_xyzs: np.ndarray, class_masks: np.ndarray):
    P = raw_instance_features.shape[0]
    assert raw_xyzs.shape[0] == P and raw_xyzs.shape[1] == 3
    labels = np.full((P,), -1, dtype='i8')
    base_label = 0
    
    for label_mask, class_id in zip(class_masks, range(class_masks.shape[0])):
        masked_raw_instance_features = raw_instance_features[label_mask]
        masked_raw_semantic_features = raw_semantic_features[label_mask]
        masked_raw_xyzs = raw_xyzs[label_mask]
        
        if masked_raw_instance_features.shape[0] == 0:
            continue
        sample_mask, _ = get_sample_mask(masked_raw_instance_features.shape[0], args.sample_num)
        masked_instance_features = feature_preprocess(masked_raw_instance_features)
        masked_semantic_features = feature_preprocess(masked_raw_semantic_features)
        masked_xyzs = xyz_preprocess(masked_raw_xyzs)
        sample_instance_features = masked_instance_features[sample_mask]
        sample_semantic_features = masked_semantic_features[sample_mask]
        sample_xyzs = masked_xyzs[sample_mask]
        masked_clusters = hybird_clustering(args, sample_instance_features, sample_semantic_features, sample_xyzs)
        if len(masked_clusters) == 0:
            continue
        masked_labels = assign_label(args, masked_clusters, masked_instance_features, masked_semantic_features, masked_xyzs)
        masked_labels = labels_postprocess(args, masked_xyzs, masked_labels)
        
        # Apply SOR filter to each cluster separately
        if args.use_sor and len(masked_raw_xyzs) > 0:
            # Get unique cluster labels (excluding -1 which is background)
            unique_cluster_ids = np.unique(masked_labels[masked_labels >= 0])
            
            for cluster_id in unique_cluster_ids:
                # Create mask for current cluster
                cluster_mask = (masked_labels == cluster_id)
                cluster_xyzs = masked_raw_xyzs[cluster_mask]
                
                if len(cluster_xyzs) > 0:
                    print(f"Applying SOR filter to cluster {cluster_id} in class {class_id} with {len(cluster_xyzs)} points...")
                    sor_mask = sor_filter_outliers(cluster_xyzs, args.sor_nb_neighbors, args.sor_std_ratio)
                    outliers_count = (~sor_mask).sum()
                    print(f"SOR filter removed {outliers_count} outliers out of {len(cluster_xyzs)} points in cluster {cluster_id}")
                    
                    # Mark outliers as background (-1)
                    masked_labels[cluster_mask] = np.where(sor_mask, cluster_id, -1)
        
        labels[label_mask] = np.where(masked_labels>=0, masked_labels + base_label, -1)
        base_label += len(masked_clusters)
    return labels

def choose_class(classes, vote):
    vote, backgound_vote = vote[:-1], vote[-1]
    if vote.max() <= backgound_vote:
        return 'background'
    return classes[vote.argmax().item()]

def assign_class(args: ClusteringConfig, cluster_labels, cameras, feature_gaussians, pipe, background_color, background_feature):
    cluster_to_class = {i.item(): np.zeros(len(args.classes)+1, dtype='i8') for i in np.unique(cluster_labels) if i>=0}
    for sample in tqdm(DataLoader(cameras, batch_size=None, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=os.cpu_count()>0, prefetch_factor=2 if os.cpu_count()>0 else None)):
        camera = sample.camera
        masks_tensor = sample.masks
        labels_tensor = sample.labels
        N, H, W = masks_tensor.shape
        if N==0:
            continue
        masks = masks_tensor.numpy()
        background_mask = ~masks.any(axis = 0)
        class_labels = labels_tensor.numpy()
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

def assign_class_semantic(args: ClusteringConfig, cluster_labels, feature_gaussians):
    lbl_feats_np = torch.load(args.segment_label_features_path, weights_only=True).numpy()
    pt_feats_np = feature_gaussians.get_semantic_features.detach().cpu().numpy() # 或者是直接用变量名
    cluster_ids_np = cluster_labels # 已经是 numpy 了
    # ==========================================
    # 2. 计算每个簇的中心特征 (Cluster Centroids)
    # ==========================================
    unique_clusters = np.unique(cluster_ids_np)
    cluster_centers = []
    valid_cluster_ids = []

    print(f"正在计算 {len(unique_clusters)} 个簇的中心特征...")

    for cid in unique_clusters:
        # 找到属于当前簇 cid 的所有点的索引
        mask = (cluster_ids_np == cid)
        
        # 取出对应的特征向量
        cluster_points = pt_feats_np[mask]
        
        # 计算均值作为该簇的代表特征
        # axis=0 表示沿着列方向求平均，结果 shape 为 (32,)
        mean_feat = np.mean(cluster_points, axis=0)
        
        cluster_centers.append(mean_feat)
        valid_cluster_ids.append(cid)

    cluster_centers = np.array(cluster_centers) # Shape: [Num_Clusters, 32]
    valid_cluster_ids = np.array(valid_cluster_ids)
    # ==========================================
    # 3. 计算相似度并分配标签 (Cosine Similarity)
    # ==========================================
    # 为了计算余弦相似度，我们需要先对向量进行归一化 (L2 Norm)
    # 归一化后：A . B = cos(theta)
    # 对 label features 进行归一化
    lbl_norm = np.linalg.norm(lbl_feats_np, axis=1, keepdims=True)
    lbl_feats_normalized = lbl_feats_np / (lbl_norm + 1e-8) # 加 1e-8 防止除零
    # 对 cluster centers 进行归一化
    ctr_norm = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    centers_normalized = cluster_centers / (ctr_norm + 1e-8)
    # 计算相似度矩阵: [Num_Clusters, 32] @ [32, 22] -> [Num_Clusters, 22]
    similarity_matrix = centers_normalized @ lbl_feats_normalized.T
    # 找到每个簇最相似的 label 索引 (axis=1 表示在每一行中找最大值的索引)
    assigned_label_indices = np.argmax(similarity_matrix, axis=1)
    max_similarity = np.max(similarity_matrix, axis=1)
    # 过滤掉相似度低于阈值的簇
    print(f'{max_similarity=}')
    valid_mask = (max_similarity >= 0.99)
    valid_cluster_ids = valid_cluster_ids[valid_mask]
    assigned_label_indices = assigned_label_indices[valid_mask]
    # ==========================================
    # 4. 结果整理
    # ==========================================
    # 创建一个字典映射: Cluster_ID -> Label_Index
    cluster_to_label_map = {
        cid: args.classes[lbl_idx.item()]
        for cid, lbl_idx in zip(valid_cluster_ids, assigned_label_indices)
    }
    return cluster_to_label_map

def output_json(args: ClusteringConfig, labels, classes, **kwargs):
    output = {}
    output['point_labels'] = labels
    instances = {str(cluster): {'class': klass} for cluster, klass in classes.items()}
    output['instances'] = instances
    for k, v in kwargs:
        output[k] = v
    with open(args.json_path,'w') as f:
        json.dump(output,f)

def clean(
    args: ClusteringConfig,
):
    if not args.clean:
        return
    if os.path.isdir(args.segment_masks_dir):
        shutil.rmtree(args.segment_masks_dir)
    if os.path.isdir(args.segment_labels_dir):
        shutil.rmtree(args.segment_labels_dir)
    if os.path.isfile(args.feature_point_cloud_path):
        os.remove(args.feature_point_cloud_path)

def build_gaussian_mask(args: ClusteringConfig, feature_gaussians):
    opacity = feature_gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
    scaling = feature_gaussians.get_scaling.detach().cpu().numpy().max(axis=1)
    return (opacity >= args.opacity_threshold) & (scaling <= args.scale_threshold)


def calc_class_masks(args: ClusteringConfig, feature_gaussians):
    label_features = torch.load(args.segment_label_features_path, weights_only=True).numpy()
    semantic_features = feature_gaussians.get_semantic_features.detach().cpu().numpy()
    P, C = semantic_features.shape
    L, D = label_features.shape
    assert C == D
    if L != len(args.classes):
        raise ValueError(f"Expected {len(args.classes)} label features, got {L}")

    similarity = semantic_features @ label_features.T  # (P, L)
    valid_gaussian_mask = build_gaussian_mask(args, feature_gaussians)
    selected_classes = set(args.selected_classes)

    # class_masks shape is [L, P]
    # gaussian i 属于 class j 当且仅当 similarity[i, j] 是该行的最大值且大于 threshold
    max_idx = similarity.argmax(axis=1)  # (P,)
    max_values = similarity.max(axis=1)  # (P,)
    class_masks = []
    for class_id, class_name in enumerate(args.classes):
        class_mask = (max_idx == class_id) & (max_values >= args.background_threshold) & valid_gaussian_mask
        if class_name not in selected_classes:
            class_mask = np.zeros(P, dtype=bool)
        class_masks.append(class_mask)
    class_masks = np.stack(class_masks, axis=0)  # (L, P)
    return class_masks

@hydra.main(config_path="configs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    app_cfg = ClusteringAppConfig(**OmegaConf.to_container(cfg, resolve=True))
    model: ModelConfig = app_cfg.model
    args: ClusteringConfig = app_cfg.clustering

    safe_state(args.quiet)
    feature_gaussians = load_model(args, model)
    class_masks = calc_class_masks(args, feature_gaussians)
    instance_features = feature_gaussians.get_instance_features.cpu().numpy()
    semantic_features = feature_gaussians.get_semantic_features.cpu().numpy()
    labels = clustering(args, instance_features, semantic_features, feature_gaussians.get_xyz.cpu().numpy(), class_masks)
    cluster_to_class = assign_class_semantic(args, labels, feature_gaussians)
    output_json(args, labels.tolist(), cluster_to_class)
    clean(args)

if __name__ == "__main__":
    main()
