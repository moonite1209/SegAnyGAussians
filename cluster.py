import logging
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

logger = logging.getLogger(__name__)

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
    """
    Generate a sampling mask.

    Args:
        total_num: Total number of points
        sample_num:
            - If 0 < sample_num < 1: sample this proportion of points
            - If >= 1: sample this exact number of points

    Returns:
        sampled_mask: boolean mask of sampled points
        actual_sample_num: actual number of sampled points
    """
    if sample_num <= 1:
        # Sample by proportion (0-1)
        n_samples = max(1, int(total_num * sample_num))
        sampled_mask = uniform_sample(total_num, n_samples)
    else:
        # Sample by exact count
        n_samples = int(sample_num)
        sampled_mask = uniform_sample(total_num, n_samples)
    return sampled_mask, sampled_mask.sum().item()

def feature_preprocess(features):
    return features
def xyz_preprocess(xyz):
    return robust_scale(xyz)

def hybird_clustering(args, instance_features, semantic_features, xyzs):
    """Hybrid clustering using instance, semantic features, and spatial coordinates."""
    # Compute distance matrices
    instance_distance_matrix = pairwise_distances(instance_features, metric='cosine')
    semantic_distance_matrix = pairwise_distances(semantic_features, metric='cosine')
    xyz_distance_matrix = pairwise_distances(xyzs, metric='euclidean')

    # Log distance statistics before normalization
    logger.debug(f"Instance distance range: [{instance_distance_matrix.min():.4f}, {instance_distance_matrix.max():.4f}]")
    logger.debug(f"Semantic distance range: [{semantic_distance_matrix.min():.4f}, {semantic_distance_matrix.max():.4f}]")
    logger.debug(f"XYZ distance range: [{xyz_distance_matrix.min():.4f}, {xyz_distance_matrix.max():.4f}]")

    # Normalize each distance matrix to [0, 1]
    instance_distance_matrix = instance_distance_matrix / instance_distance_matrix.max()
    semantic_distance_matrix = semantic_distance_matrix / semantic_distance_matrix.max()
    xyz_distance_matrix = xyz_distance_matrix / xyz_distance_matrix.max()

    distance_matrix = (args.instance_feature_ratio * instance_distance_matrix +
                      args.semantic_feature_ratio * semantic_distance_matrix +
                      args.xyz_feature_ratio * xyz_distance_matrix)

    logger.debug(f"Weighted distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    logger.debug(f"Distance weights - instance: {args.instance_feature_ratio}, semantic: {args.semantic_feature_ratio}, xyz: {args.xyz_feature_ratio}")

    clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster=False, metric='precomputed', n_jobs=-1)
    sample_labels = clusterer.fit_predict(distance_matrix)

    n_clusters = len(sample_labels[sample_labels >= 0])
    n_noise = len(sample_labels[sample_labels == -1])
    logger.info(f"Clustering result: {n_clusters} points, {n_noise} noise points out of {len(sample_labels)} total samples")

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
    """Assign cluster labels to all points based on feature similarities."""
    P, C_instance = instance_features.shape
    P, C_semantic = semantic_features.shape
    P, D = xyzs.shape
    label = np.array([cluster['label'] for cluster in clusters])

    logger.debug(f"Assigning labels to {P} points based on {len(clusters)} clusters")

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

    n_assigned = is_valid.sum()
    n_rejected = (~is_valid).sum()
    logger.debug(f"Label assignment: {n_assigned} points assigned, {n_rejected} points rejected (threshold={args.instance_threshold})")

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

def apply_sor_to_clusters(labels: np.ndarray, xyzs: np.ndarray, snb_neighbors: int, std_ratio: float):
    """
    Apply Statistical Outlier Removal (SOR) filter to each cluster in the points

    Args:
        labels: numpy array of cluster IDs for points
        xyzs: numpy array of 3D coordinates for points
        snb_neighbors: number of neighbors to consider for SOR filter
        std_ratio: standard deviation ratio threshold for SOR filter

    Returns:
        updated_labels: numpy array with outliers marked as background (-1)
    """
    # Get unique cluster labels (excluding -1 which is background)
    unique_cluster_ids = np.unique(labels[labels >= 0])

    updated_labels = labels.copy()

    for cluster_id in unique_cluster_ids:
        # Create mask for current cluster
        cluster_mask = (labels == cluster_id)
        cluster_xyzs = xyzs[cluster_mask]

        if len(cluster_xyzs) > 0:
            logger.info(f"Applying SOR filter to cluster {cluster_id} with {len(cluster_xyzs)} points...")
            inlier_mask = sor_filter_outliers(cluster_xyzs, snb_neighbors, std_ratio)
            outliers_count = (~inlier_mask).sum()
            logger.info(f"SOR filter removed {outliers_count} outliers out of {len(cluster_xyzs)} points in cluster {cluster_id}")

            # Mark outliers as background (-1)
            updated_labels[cluster_mask] = np.where(inlier_mask, cluster_id, -1)

    return updated_labels

def filter_large_gaussians(labels: np.ndarray, scales: np.ndarray, scale_threshold: float = None,
                           use_percentile: bool = True, percentile: float = 95.0):
    """
    Filter large Gaussians based on their scales.

    Args:
        labels: numpy array of cluster IDs for points
        scales: numpy array of shape (N, 3) representing scale values for each Gaussian
        scale_threshold: absolute threshold for scale filtering. If None, uses percentile-based filtering
        use_percentile: if True, use percentile-based filtering within each cluster
        percentile: percentile threshold (0-100) for cluster-based filtering

    Returns:
        updated_labels: numpy array with large Gaussians marked as background (-1)
    """
    # Compute max scale for each Gaussian (take the maximum of the 3 scale dimensions)
    max_scales = np.max(scales, axis=1)

    # Branch 1: Percentile mode - need to process each cluster separately
    if use_percentile and scale_threshold is None:
        updated_labels = labels.copy()
        unique_cluster_ids = np.unique(labels[labels >= 0])

        for cluster_id in unique_cluster_ids:
            cluster_mask = (labels == cluster_id)
            cluster_max_scales = max_scales[cluster_mask]

            if len(cluster_max_scales) == 0:
                continue

            # Compute cluster-specific threshold
            scale_threshold = np.percentile(cluster_max_scales, percentile)
            inlier_mask = cluster_max_scales <= scale_threshold
            outliers_count = (~inlier_mask).sum()

            if outliers_count > 0:
                logger.info(f"Filtering {outliers_count} large Gaussians out of {len(cluster_max_scales)} "
                           f"in cluster {cluster_id} (threshold: {scale_threshold:.4f})")
                # Mark large Gaussians as background (-1)
                updated_labels[cluster_mask] = np.where(inlier_mask, cluster_id, -1)

        return updated_labels

    # Branch 2: Absolute threshold mode - global filtering (more efficient)
    else:
        if scale_threshold is None:
            logger.warning("scale_threshold is None and use_percentile is False. Skipping filter.")
            return labels

        # Always create a copy to preserve SOR filtering results
        updated_labels = labels.copy()
        inlier_mask = max_scales <= scale_threshold
        outliers_count = (~inlier_mask).sum()

        if outliers_count > 0:
            logger.info(f"Filtering {outliers_count} large Gaussians out of {len(max_scales)} "
                       f"(global threshold: {scale_threshold:.4f})")

            # Only filter labeled points, preserve background (-1)
            valid_mask = (labels >= 0)
            updated_labels[valid_mask] = np.where(
                inlier_mask[valid_mask],
                labels[valid_mask],
                -1
            )

        return updated_labels

def clustering(args, raw_instance_features: np.ndarray, raw_semantic_features: np.ndarray, raw_xyzs: np.ndarray, class_masks: np.ndarray):
    """Main clustering function processing each class separately."""
    P, C_instance = raw_instance_features.shape
    P, C_semantic = raw_semantic_features.shape
    assert raw_xyzs.shape[0] == P and raw_xyzs.shape[1] == 3
    labels = np.full((P,), -1, dtype='i8')
    base_label = 0

    logger.info(f"Starting clustering for {P} points across {class_masks.shape[0]} classes")

    for class_id, label_mask in enumerate(class_masks):
        masked_raw_instance_features = raw_instance_features[label_mask]
        masked_raw_semantic_features = raw_semantic_features[label_mask]
        masked_raw_xyzs = raw_xyzs[label_mask]

        n_points_in_class = masked_raw_instance_features.shape[0]
        if n_points_in_class == 0:
            continue

        logger.info(f"Processing class {class_id} ({args.classes[class_id] if hasattr(args, 'classes') and class_id < len(args.classes) else class_id}): {n_points_in_class} points")

        sample_mask, sample_num = get_sample_mask(masked_raw_instance_features.shape[0], args.sample_num)
        masked_instance_features = feature_preprocess(masked_raw_instance_features)
        masked_semantic_features = feature_preprocess(masked_raw_semantic_features)
        masked_xyzs = xyz_preprocess(masked_raw_xyzs)
        sample_instance_features = masked_instance_features[sample_mask]
        sample_semantic_features = masked_semantic_features[sample_mask]
        sample_xyzs = masked_xyzs[sample_mask]

        logger.debug(f"Sampling {sample_num} points for clustering")

        masked_clusters = hybird_clustering(args, sample_instance_features, sample_semantic_features, sample_xyzs)
        if len(masked_clusters) == 0:
            logger.warning(f"No clusters found for class {class_id}, skipping")
            continue

        logger.info(f"Found {len(masked_clusters)} clusters for class {class_id}")

        masked_labels = assign_label(args, masked_clusters, masked_instance_features, masked_semantic_features, masked_xyzs)
        n_labeled = (masked_labels >= 0).sum()
        logger.debug(f"Assigned labels to {n_labeled} points out of {len(masked_labels)}")

        masked_labels = labels_postprocess(args, masked_xyzs, masked_labels)
        labels[label_mask] = np.where(masked_labels>=0, masked_labels + base_label, -1)
        base_label += len(masked_clusters)

    total_clusters = base_label
    total_labeled = (labels >= 0).sum()
    logger.info(f"Clustering complete: {total_clusters} total clusters, {total_labeled}/{P} points labeled")

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
    pt_feats_np = feature_gaussians.get_semantic_features.detach().cpu().numpy() # 或者直接使用变量名
    cluster_ids_np = cluster_labels # 已经是 numpy 了
    # ==========================================
    # 2. 计算每个簇的中心特征 (Cluster Centroids)
    # ==========================================
    unique_clusters = np.unique(cluster_ids_np)
    cluster_centers = []
    valid_cluster_ids = []

    logger.info(f"Computing centroid features for {len(unique_clusters)} clusters...")

    for cid in unique_clusters:
        # 找到属于当前簇 cid 的所有点的索引
        mask = (cluster_ids_np == cid)

        # 提取对应的特征向量
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
    # Filter clusters with similarity below threshold
    logger.debug(f"Max similarity per cluster: {max_similarity}")
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

def get_bbox(labels, xyz):
    """
    Compute 3D oriented bounding boxes for each instance cluster.
    Projects points to X-Z plane and uses trimesh to compute oriented 2D bounds,
    then extends to 3D by including Y range.

    Args:
        labels: instance labels for each point
        xyz: point coordinates (N, 3)

    Returns:
        dict mapping instance_id to flattened bbox corners (24 floats)
    """
    from trimesh.bounds import oriented_bounds_2D

    bbox = {}
    for instance_id in np.unique(labels):
        if instance_id < 0:
            continue
        instance_xyz = xyz[(labels == instance_id)]
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

def calc_bbox(args, labels, xyzs, scales):
    """Calculate bounding boxes with optional filtering."""
    filtered_labels = labels.copy()
    n_original_clusters = len(labels[labels >= 0])

    logger.info(f"Computing bounding boxes for {n_original_clusters} clusters")

    # Apply SOR filter
    if args.use_sor:
        logger.info("Applying SOR (Statistical Outlier Removal) filter...")
        filtered_labels = apply_sor_to_clusters(filtered_labels, xyzs, args.sor_nb_neighbors, args.sor_std_ratio)
        n_after_sor = len(filtered_labels[filtered_labels >= 0])
        logger.info(f"After SOR: {n_after_sor}/{n_original_clusters} clusters remaining")

    # Apply Big Gaussian filter
    if args.use_scale_filter:
        filter_mode = "percentile" if args.use_scale_percentile else "absolute threshold"
        logger.info(f"Applying large Gaussian filter ({filter_mode} mode, percentile={args.scale_percentile if args.use_scale_percentile else 'N/A'})...")
        filtered_labels = filter_large_gaussians(
            filtered_labels,
            scales,
            scale_threshold=args.scale_threshold if hasattr(args, 'scale_threshold') else None,
            use_percentile=args.use_scale_percentile if hasattr(args, 'use_scale_percentile') else True,
            percentile=args.scale_percentile if hasattr(args, 'scale_percentile') else 95.0
        )
        n_after_scale = len(filtered_labels[filtered_labels >= 0])
        logger.info(f"After scale filter: {n_after_scale}/{n_after_sor} clusters remaining")

    bboxs = get_bbox(filtered_labels, xyzs)
    logger.info(f"Generated {len(bboxs)} bounding boxes")

    return bboxs, filtered_labels

def output_json(args, labels, classes, bboxs):
    """
    Output clustering results to JSON file.

    Args:
        args: configuration arguments containing json_path and selected_classes
        labels: point labels (numpy array or list), original unfiltered labels
        classes: dictionary mapping cluster_id to class name
        bboxs: dictionary mapping cluster_id to bounding box (24 floats: 8 corners x 3 coords)
    """
    logger.info(f"Saving output to {args.json_path}")

    output = {}
    output['point_labels'] = labels

    # Combine bbox and class information
    instances = {
        str(cid): {
            'class': klass,
            **({'bbox': bboxs[cid]} if cid in bboxs else {})
        }
        for cid, klass in classes.items()
    }

    # Filter instances to only include selected_classes
    output['instances'] = {
        k: v for k, v in instances.items()
        if v.get('class') in args.selected_classes
    }

    logger.info(f"Output contains {len(output['instances'])} instances (filtered from {len(instances)} total)")

    with open(args.json_path, 'w') as f:
        json.dump(output, f)

    logger.info(f"Successfully saved results to {args.json_path}")

def clean(args, dataset):
    if os.path.isdir(dataset.masks_path):
        shutil.rmtree(dataset.masks_path)
    if os.path.isdir(dataset.labels_path):
        shutil.rmtree(dataset.labels_path)
    if os.path.isfile(args.feature_point_cloud_path):
        os.remove(args.feature_point_cloud_path)

def calc_class_masks(args, dataset, feature_gaussians):
    """Calculate class masks based on semantic feature similarity."""
    label_features = torch.load(os.path.join(dataset.labels_path, 'label_features.pt'), weights_only=True).numpy()
    P, C = feature_gaussians.shape
    L, D = label_features.shape
    assert C == D

    logger.debug(f"Computing class masks: {P} points, {L} classes, {C} feature dimensions")

    similarity = (feature_gaussians @ label_features.T)  # (P, L)
    # class_masks shape is [L, P]
    # gaussian i belongs to class j when similarity[i, j] is max and >= threshold
    max_idx = similarity.argmax(axis=1)  # (P,)
    max_values = similarity.max(axis=1)  # (P,)

    # Log similarity statistics
    logger.debug(f"Similarity statistics - min: {max_values.min():.4f}, max: {max_values.max():.4f}, mean: {max_values.mean():.4f}")

    class_masks = []
    class_counts = []
    for class_id in range(L):
        class_mask = (max_idx == class_id) & (max_values >= 0.25)
        class_masks.append(class_mask)
        class_counts.append(class_mask.sum())

    # Log class distribution
    for class_id, count in enumerate(class_counts):
        if count > 0:
            class_name = args.classes[class_id] if hasattr(args, 'classes') and class_id < len(args.classes) else f"class_{class_id}"
            logger.debug(f"Class {class_id} ({class_name}): {count} points ({100*count/P:.2f}%)")

    class_masks = np.stack(class_masks, axis=0)  # (L, P)
    n_assigned = (class_masks.any(axis=0)).sum()
    logger.info(f"Class masks: {n_assigned}/{P} points assigned to classes ({100*n_assigned/P:.2f}%)")

    return class_masks

def write_progress(progress_path, value):
    if progress_path:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, 'w') as f:
            f.write(value)

@hydra.main(config_path="configs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    """Main clustering pipeline."""
    model = cfg.model
    dataset = cfg.dataset
    pipe = cfg.pipe
    args = cfg.clustering

    logger.info("=" * 80)
    logger.info("Starting SAGA clustering pipeline")
    logger.info("=" * 80)

    # Verify that the sum of ratios is approximately 1
    total_feature_ratio = args.instance_feature_ratio + args.semantic_feature_ratio + args.xyz_feature_ratio
    if not np.isclose(total_feature_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Feature ratios must sum to 1.0, but got {total_feature_ratio}")

    logger.info(f"Feature weights - instance: {args.instance_feature_ratio}, semantic: {args.semantic_feature_ratio}, xyz: {args.xyz_feature_ratio}")
    logger.info(f"Sample size: {args.sample_num}")
    logger.info(f"SOR filter: {args.use_sor}, Scale filter: {args.use_scale_filter}")

    write_progress(args.progress_path, '0')

    safe_state(args.quiet)
    feature_gaussians, background_color, background_feature = load_model(args, model)
    cameras = load_cameras(dataset)
    n_cameras = len(cameras)
    n_gaussians = len(feature_gaussians.get_xyz)

    logger.info(f"Loaded {n_cameras} cameras and {n_gaussians} Gaussians")

    # Write progress: data loaded (0-10%)
    write_progress(args.progress_path, '10')

    # Calculate class masks based on semantic feature similarity
    class_masks = calc_class_masks(args, dataset, feature_gaussians.get_semantic_features.cpu().numpy())
    logger.info(f"Generated class masks for {class_masks.shape[0]} classes")

    # Extract features and coordinates from feature gaussians
    instance_features = feature_gaussians.get_instance_features.cpu().numpy()
    semantic_features = feature_gaussians.get_semantic_features.cpu().numpy()
    xyzs = feature_gaussians.get_xyz.cpu().numpy()
    scales = feature_gaussians.get_scaling.cpu().numpy()

    # Perform hybrid clustering
    labels = clustering(args, instance_features, semantic_features, xyzs, class_masks)

    # Write progress: clustering done (50-60%)
    write_progress(args.progress_path, '60')

    # Assign class labels to each cluster
    cluster_to_class = assign_class_semantic(args, labels, cameras, feature_gaussians, pipe, background_color, background_feature)
    logger.info(f"Assigned class labels to {len(cluster_to_class)} clusters")

    bboxs, bbox_labels = calc_bbox(args, labels, xyzs, scales)

    # Write progress: class assignment done (90-95%)
    write_progress(args.progress_path, '95')

    # Output clustering results to JSON file
    # labels = apply_sor_to_clusters(labels, xyzs, 30, 2)
    # labels = filter_large_gaussians(labels, scales, use_percentile=True, percentile=99.0)
    output_json(args, labels.tolist(), cluster_to_class, bboxs)
    if args.clean:
        clean(args, dataset)

    # Write progress: all done (100%)
    write_progress(args.progress_path, '100')

    logger.info("=" * 80)
    logger.info("Clustering pipeline completed successfully!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()