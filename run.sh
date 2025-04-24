#!/bin/bash

for base_path in "$@"; do
    echo "training: $base_path"
    images_path="${base_path}/fastRecon/dense/sparse/0/images/"
    sparse_path="${base_path}/fastRecon/dense/sparse/0/"
    point_cloud_path="${base_path}/output_models/point_cloud/iteration_30000/point_cloud.ply"

    masks_path="${base_path}/masks"
    labels_path="${base_path}/labels"

    contrastive_feature_point_cloud_path="${base_path}/contrastive_feature_point_cloud.ply"

    json_path="${base_path}/output.json"
    progress_path="${base_path}/progress"

    sam_checkpoint_path="../weights/sam_vit_h_4b8939.pth"
    groundingdino_checkpoint_path="../weights/groundingdino_swint_ogc.pth"
    groundingdino_config_path="../weights/GroundingDINO_SwinT_OGC.py"

    python grounded_SAM_masks.py --progress_path $progress_path --images_path $images_path --masks_path $masks_path --labels_path $labels_path --sam_checkpoint_path $sam_checkpoint_path --groundingdino_checkpoint_path $groundingdino_checkpoint_path --groundingdino_config_path $groundingdino_config_path --downsample 1 && \
    python train_contrastive_feature.py --progress_path $progress_path --sh_degree 0 --feature_dim 32 --images_path $images_path --sparse_path $sparse_path --masks_path $masks_path --point_cloud_path $point_cloud_path --contrastive_feature_point_cloud_path $contrastive_feature_point_cloud_path --num_sampled_rays 1000 && \
    python postprocess.py --progress_path $progress_path --sh_degree 0 --feature_dim 32 --images_path $images_path --sparse_path $sparse_path --masks_path $masks_path --labels_path $labels_path --point_cloud_path $point_cloud_path --contrastive_feature_point_cloud_path $contrastive_feature_point_cloud_path --json_path $json_path
done