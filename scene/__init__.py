#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from scene.feature_gaussian_model import FeatureGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel
    feature_gaussians : FeatureGaussianModel

    # target: feature, seg, scene
    def __init__(self, args : ModelParams, gaussians : GaussianModel=None, shuffle=True, resolution_scales=[1.0], sample_rate = 1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.sparse_path)):
            print(f"Allow Camera Principle Point Shift: {args.allow_principle_point_shift}")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.sparse_path, args.images, args.eval, sample_rate = sample_rate, args=args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Lerf data set!")
            scene_info = sceneLoadTypeCallbacks["Lerf"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not os.path.exists(os.path.join(self.model_path, "input.ply")):
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
        if not os.path.exists(os.path.join(self.model_path, "cameras.json")):
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class FeatureScene:
    feature_gaussians : FeatureGaussianModel

    def __init__(self, args : ModelParams, feature_gaussians: FeatureGaussianModel=None, shuffle=True, resolution_scales=[1.0], sample_rate = 1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.feature_gaussians = feature_gaussians
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.sparse_path)):
        # used for testing lerf transforms,json
            print(f"Allow Camera Principle Point Shift: {args.allow_principle_point_shift}")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.sparse_path, args.images, args.eval, sample_rate = sample_rate, args=args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]