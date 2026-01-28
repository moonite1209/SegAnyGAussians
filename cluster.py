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
import open3d as o3d

def uniform_sample(N, n_samples):
    selected_indices = np.random.permutation(N)[:n_samples]
    mask = np.zeros(N, dtype='b1')
    mask[selected_indices] = True
    return mask

def load_model(args, model):
    feature_gaussians = FeatureGaussianModel(model.sh_degree, model.instance_feature_dim, model.semantic_feature_dim)
    feature_gaussians.load_ply(args.feature_point_cloud_path)
    feature_gaussians.eval()
    background_color = torch.tensor([1.]*3 if model.white_background else [0.]*3, dtype=torch.float32, device="cuda")
    background_feature = torch.tensor([0.]*model.instance_feature_dim, dtype=torch.float32, device="cuda")
    return feature_gaussians, background_color, background_feature

def load_cameras(dataset):
    scene = FeatureScene(dataset)
    return scene.getCameraDataset()

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

def hybird_clustering(args, instance_features, semantic_features, xyzs):
    
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

def assign_label(args, clusters, instance_features, semantic_features, xyzs):
    P, C_instance = instance_features.shape
    P, C_semantic = semantic_features.shape
    P, D = xyzs.shape
    label = np.array([cluster['label'] for cluster in clusters])
    
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

def labels_postprocess(args, xyzs, labels):
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

def clustering(args, raw_instance_features: np.ndarray, raw_semantic_features: np.ndarray, raw_xyzs: np.ndarray, class_masks: np.ndarray):
    P, C_instance = raw_instance_features.shape
    P, C_semantic = raw_semantic_features.shape
    assert raw_xyzs.shape[0] == P and raw_xyzs.shape[1] == 3
    labels = np.full((P,), -1, dtype='i8')
    base_label = 0
    
    for label_mask, class_id in zip(class_masks, range(class_masks.shape[0])):
        masked_raw_instance_features = raw_instance_features[label_mask]
        masked_raw_semantic_features = raw_semantic_features[label_mask]
        masked_raw_xyzs = raw_xyzs[label_mask]

        if masked_raw_instance_features.shape[0] == 0:
            continue
        sample_mask, sample_num = get_sample_mask(masked_raw_instance_features.shape[0], args.sample_num)
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

    # sample_mask, sample_num = get_sample_mask(P, args.sample_num)
    # features = feature_preprocess(raw_features)
    # xyzs = xyz_preprocess(raw_xyzs)
    # sample_features = features[sample_mask]
    # sample_xyzs = xyzs[sample_mask]
    # clusters = hybird_clustering(args, sample_features, sample_xyzs)
    # labels = assign_label(args, clusters, features, xyzs)
    # labels = labels_postprocess(args, xyzs, labels)
    # return labels

def choose_class(classes, vote):
    vote, backgound_vote = vote[:-1], vote[-1]
    if vote.max() <= backgound_vote:
        return 'background'
    return classes[vote.argmax().item()]

def assign_class(args, cluster_labels, cameras, feature_gaussians, pipe, background_color, background_feature):
    cluster_to_class = {i.item(): np.zeros(len(args.classes)+1, dtype='i8') for i in np.unique(cluster_labels) if i>=0}
    for camera in tqdm(DataLoader(cameras, batch_size=None, shuffle=False, num_workers=os.cpu_count())):
        N,H,W = camera.original_masks.shape
        if N==0:
            continue
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

def assign_class_semantic(args, cluster_labels, cameras, feature_gaussians, pipe, background_color, background_feature):
    lbl_feats_np = torch.load(args.features_path, weights_only=True).numpy()
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

def get_bbox(labels, xyz, is_big_gaussian):
    """
    Compute 3D oriented bounding boxes for each instance cluster.
    Projects points to X-Z plane and uses trimesh to compute oriented 2D bounds,
    then extends to 3D by including Y range.

    Args:
        labels: instance labels for each point
        xyz: point coordinates (N, 3)
        is_big_gaussian: boolean mask filtering out large gaussians

    Returns:
        dict mapping instance_id to flattened bbox corners (24 floats)
    """
    from trimesh.bounds import oriented_bounds_2D

    bbox = {}
    for instance_id in np.unique(labels):
        if instance_id < 0:
            continue
        instance_xyz = xyz[(labels == instance_id) & ~is_big_gaussian]
        points_3d = instance_xyz

        # Skip if too few points
        if len(points_3d) < 3:
            continue

        # --- 1. Project to X-Z plane (ignore Y) ---
        N, D = points_3d.shape
        points_2d = points_3d[:, [0, 2]]  # shape: (N, 2), X and Z coordinates

        # --- 2. Compute 2D oriented bounding box using trimesh ---
        # oriented_bounds_2D returns a transform matrix that transforms points
        # so their AABB center is at origin
        transform_2d, rectangle_extents_2d = oriented_bounds_2D(points_2d)
        # transform_2d: (3, 3) 2D homogeneous transform matrix
        # rectangle_extents_2d: (2,) [width, height] in transformed 2D space

        # --- 3. Extend 2D transform to 3D ---
        transform_3d = np.eye(4)  # Initialize as identity matrix

        # Copy rotation and translation from 2D transform to 3D transform
        # transform_2d is:
        # [ R_xx  R_xz  tx ]
        # [ R_zx  R_zz  tz ]
        # [  0     0    1 ]
        transform_3d[0, 0] = transform_2d[0, 0]  # R_xx
        transform_3d[0, 2] = transform_2d[0, 1]  # R_xz
        transform_3d[0, 3] = transform_2d[0, 2]  # tx

        transform_3d[2, 0] = transform_2d[1, 0]  # R_zx
        transform_3d[2, 2] = transform_2d[1, 1]  # R_zz
        transform_3d[2, 3] = transform_2d[1, 2]  # tz

        # Y axis remains unchanged: transform_3d[1,1] = 1, others are 0 (already set by eye(4))

        # --- 4. Apply 3D transform to "align" points ---
        points_3d_hom = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
        points_3d_transformed = (transform_3d @ points_3d_hom.T).T  # (N, 4)
        points_3d_transformed = points_3d_transformed[:, :3]  # Remove homogeneous dimension (N, 3)

        # --- 5. Compute AABB of transformed point cloud ---
        aabb_min = points_3d_transformed.min(axis=0)  # (3,)
        aabb_max = points_3d_transformed.max(axis=0)  # (3,)

        # --- 6. Build 8 corner points in local (transformed) space ---
        half_extents_xz = rectangle_extents_2d / 2.0
        corners_local = np.array([
            [ half_extents_xz[0],  aabb_max[1],  half_extents_xz[1]],
            [ half_extents_xz[0],  aabb_max[1], -half_extents_xz[1]],
            [ half_extents_xz[0],  aabb_min[1], -half_extents_xz[1]],
            [ half_extents_xz[0],  aabb_min[1],  half_extents_xz[1]],
            [-half_extents_xz[0],  aabb_max[1],  half_extents_xz[1]],
            [-half_extents_xz[0],  aabb_max[1], -half_extents_xz[1]],
            [-half_extents_xz[0],  aabb_min[1], -half_extents_xz[1]],
            [-half_extents_xz[0],  aabb_min[1],  half_extents_xz[1]]
        ])  # (8, 3)

        # --- 7. Transform local corners back to world coordinates ---
        transform_3d_inv = np.linalg.inv(transform_3d)
        corners_local_hom = np.hstack([corners_local, np.ones((8, 1))])  # (8, 4)
        bbox_corners_world_hom = (transform_3d_inv @ corners_local_hom.T).T  # (8, 4)
        bbox_corners_world = bbox_corners_world_hom[:, :3]  # (8, 3)
        bbox[instance_id] = bbox_corners_world.flatten().tolist()
    return bbox

def output_json(args, labels, classes, xyz, is_big_gaussian):
    output = {}
    output['point_labels'] = labels
    # Compute bounding boxes for each instance
    bbox = get_bbox(labels, xyz, is_big_gaussian)
    # Combine bbox and class information
    instances = {}
    for cluster_id, klass in classes.items():
        cluster_key = str(cluster_id)
        if cluster_id in bbox:
            instances[cluster_key] = {
                'bbox': bbox[cluster_id],
                'class': klass
            }
        else:
            instances[cluster_key] = {'class': klass}
    # Filter instances to only include selected_classes (like old version)
    output['instances'] = {k: v for k, v in instances.items() if v.get('class') in args.selected_classes}
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

def calc_class_masks(args, dataset, feature_gaussians):
    label_features = torch.load(os.path.join(dataset.labels_path, 'label_features.pt'), weights_only=True).numpy()
    P, C = feature_gaussians.shape
    L, D = label_features.shape
    assert C == D
    similarity = (feature_gaussians @ label_features.T)  # (P, L)
    # class_masks shape is [L, P]
    # gaussian i 属于 class j 当且仅当 similarity[i, j] 是该行的最大值且大于 threshold
    max_idx = similarity.argmax(axis=1)  # (P,)
    max_values = similarity.max(axis=1)  # (P,)
    class_masks = []
    for class_id in range(L):
        class_mask = (max_idx == class_id) & (max_values >= 0.25)
        class_masks.append(class_mask)
    class_masks = np.stack(class_masks, axis=0)  # (L, P)
    return class_masks

@hydra.main(config_path="configs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    model = cfg.model
    dataset = cfg.dataset
    pipe = cfg.pipe
    args = cfg.clustering

    # Verify that the sum of ratios is approximately 1
    total_feature_ratio = args.instance_feature_ratio + args.semantic_feature_ratio + args.xyz_feature_ratio
    if not np.isclose(total_feature_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Feature ratios must sum to 1.0, but got {total_feature_ratio}")

    # Create progress file directory if needed
    if args.progress_path:
        os.makedirs(os.path.dirname(args.progress_path), exist_ok=True)
        with open(args.progress_path, 'w') as f:
            f.write('0')

    safe_state(args.quiet)
    feature_gaussians, background_color, background_feature = load_model(args, model)
    cameras = load_cameras(dataset)

    # Write progress: data loaded (0-10%)
    if args.progress_path:
        with open(args.progress_path, 'w') as f:
            f.write('10')

    class_masks = calc_class_masks(args, dataset, feature_gaussians.get_semantic_features.cpu().numpy())
    instance_features = feature_gaussians.get_instance_features.cpu().numpy()
    semantic_features = feature_gaussians.get_semantic_features.cpu().numpy()
    xyz = feature_gaussians.get_xyz.cpu().numpy()

    # Compute is_big_gaussian filter (like old version)
    point_scales = feature_gaussians.get_scaling.detach().cpu().numpy()
    is_big_gaussian = point_scales.max(axis=-1) > (np.median(point_scales.max(axis=-1)) * args.scale_threshold)

    # features = feature_gaussians.get_semantic_features.cpu().numpy()
    labels = clustering(args, instance_features, semantic_features, xyz, class_masks)

    # Write progress: clustering done (50-60%)
    if args.progress_path:
        with open(args.progress_path, 'w') as f:
            f.write('60')

    cluster_to_class = assign_class_semantic(args, labels, cameras, feature_gaussians, pipe, background_color, background_feature)

    # Write progress: class assignment done (90-95%)
    if args.progress_path:
        with open(args.progress_path, 'w') as f:
            f.write('95')

    output_json(args, labels.tolist(), cluster_to_class, xyz, is_big_gaussian)
    clean(args)

    # Write progress: all done (100%)
    if args.progress_path:
        with open(args.progress_path, 'w') as f:
            f.write('100')

if __name__ == "__main__":
    main()