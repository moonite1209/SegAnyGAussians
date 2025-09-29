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

import torch
import torch.nn.functional as F
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image
import struct

WARNED = True
def read_dmb_file(file_path, is_confidence=False):
    """
    读取DMB文件，基于depth.cpp和conf.cpp中的格式
    
    参数:
        file_path: DMB文件路径
        is_confidence: 是否为置信度文件（使用uint8_t而非float）
        
    返回:
        numpy数组形式的深度/置信度数据，如果失败返回None
    """
    try:
        with open(file_path, 'rb') as f:
            # 读取文件头信息
            type_val = struct.unpack('<i', f.read(4))[0]  # int32_t type
            h = struct.unpack('<i', f.read(4))[0]         # int32_t h (height)
            w = struct.unpack('<i', f.read(4))[0]         # int32_t w (width)
            nb = struct.unpack('<i', f.read(4))[0]        # int32_t nb (channels)
            
            # print(f"DMB文件信息: type={type_val}, 高度={h}, 宽度={w}, 通道数={nb}")
            
            # 检查类型是否正确
            if type_val != 1:
                print(f"错误：不支持的DMB文件类型: {type_val}")
                return None
            
            # 计算数据大小
            data_size = h * w * nb
            # 使用frombuffer加速解析
            if is_confidence:
                raw = f.read(data_size)
                data_array = np.frombuffer(raw, dtype=np.uint8, count=data_size)
                data_array = data_array.reshape(h, w, nb).astype(np.float32)
            else:
                raw = f.read(data_size * 4)
                data_array = np.frombuffer(raw, dtype=np.float32, count=data_size)
                data_array = data_array.reshape(h, w, nb)
            
            # 如果只有一个通道，去掉最后一维
            if nb == 1:
                data_array = data_array.squeeze()
            
            return data_array
            
    except Exception as e:
        print(f"读取DMB文件失败 {file_path}: {e}")
        return None
    
def scale_intrinsics(intrinsics: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    """
    将相机内参从源分辨率缩放到目标分辨率。
    """
    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    K = intrinsics.copy().astype(np.float32)
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy
    return K
    
def loadCam(resolution, id, cam_info, resolution_scale):
    image = Image.open(cam_info.image_path)

    orig_w, orig_h = image.size
    if resolution in [1, 2, 4, 8]:
        scale = resolution * resolution_scale
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution
        scale = float(global_down) * float(resolution_scale)
    resized_w, resized_h = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, (resized_w, resized_h))

    gt_image = resized_image_rgb[:3, ...]
    gt_alpha_mask = None

    if resized_image_rgb.shape[1] == 4:
        gt_alpha_mask = resized_image_rgb[3:4, ...]

    if cam_info.masks_path:
        masks = torch.load(cam_info.masks_path, weights_only=True)
        masks_float = masks.float()
        resized_masks_float = F.interpolate(
            masks_float.unsqueeze(1),
            size=(resized_h, resized_w),  # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        resized_masks = (resized_masks_float > 0.5).bool()
        masks = resized_masks
    else:
        masks = None

    if cam_info.labels_path:
        labels = torch.load(cam_info.labels_path, weights_only=True)
    else:
        labels = None

    # 读取深度与置信度
    if cam_info.depth_path:
        depth_map = torch.from_numpy(read_dmb_file(cam_info.depth_path, is_confidence=False).copy())[None, ...]
        resized_depth_map = F.interpolate(
            depth_map.unsqueeze(0),
            size=(resized_h, resized_w),  # (H, W)
            mode='nearest',
        ).squeeze(0)
        depth_map = resized_depth_map
    else:
        depth_map = None
    if cam_info.confidence_path:
        conf_map = torch.from_numpy(read_dmb_file(cam_info.confidence_path, is_confidence=True).copy())[None, ...]
        resized_conf_map = F.interpolate(
            conf_map.unsqueeze(0),
            size=(resized_h, resized_w),  # (H, W)
            mode='nearest',
        ).squeeze(0)
        conf_map = resized_conf_map
    else:
        conf_map = None
    # K缩放至深度图尺寸
    depth_h, depth_w = depth_map.shape[:2]

    # K = _camera_to_K(cam)
    # src_w, src_h = int(cam['width']), int(cam['height'])
    # K_scaled = scale_intrinsics(K, src_w, src_h, depth_w, depth_h)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=gt_alpha_mask,
                  image_name=cam_info.image_name, cx=cam_info.cx, cy=cam_info.cy, masks = resized_masks, labels = labels, depth_map=depth_map, confidence_map=conf_map, uid=id)

# def loadCam(emmm):

#     args, id, cam_info, resolution_scale = emmm
#     orig_w, orig_h = cam_info.image.size

#     if args.resolution in [1, 2, 4, 8]:
#         resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 1600:
#                 # global WARNED
#                 # if not WARNED:
#                 #     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                 #         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                 #     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = float(global_down) * float(resolution_scale)
#         resolution = (int(orig_w / scale), int(orig_h / scale))

#     resized_image_rgb = PILtoTorch(cam_info.image, resolution)

#     gt_image = resized_image_rgb[:3, ...]
#     loaded_mask = None

#     if resized_image_rgb.shape[1] == 4:
#         loaded_mask = resized_image_rgb[3:4, ...]
#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
#                   image=gt_image, gt_alpha_mask=loaded_mask,
#                   image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, resolution):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(resolution, id, c, resolution_scale))
    return camera_list


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cam_infos, resolution_scale, args):
        self.cam_infos = cam_infos
        self.length = len(cam_infos)
def cameraDataset_from_camInfos(cam_infos, resolution_scale, args):
    return CameraDataset(cam_infos, resolution_scale, args)


# from multiprocessing import Pool
# camera_list = []

    # for id, c in enumerate(cam_infos):
    #     camera_list.append(loadCam(args, id, c, resolution_scale))
# import torch
# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    

#     ctx = torch.multiprocessing.get_context("spawn")
#     print(torch.multiprocessing.cpu_count(), "cpu count")
#     pool = ctx.Pool(10)

#     camera_list = []
#     for id, c in enumerate(cam_infos):
#         res = pool.apply_async(loadCam, args=(args, id, c, resolution_scale))
#         camera_list.append(res)
#         print(id, "?")
#     pool.close()
#     pool.join()

#     camera_list = [i.get() for i in camera_list]

#     return camera_list

# import torch
# def cameraList_from_camInfos(cam_infos, resolution_scale, args):

#     ctx = torch.multiprocessing.get_context("spawn")
#     print(torch.multiprocessing.cpu_count(), "cpu count")
#     pool = ctx.Pool(15)

#     # camera_list = []
#     tmp = [(args, id, c, resolution_scale) for id, c in enumerate(cam_infos)]

#     camera_list = pool.map(loadCam, tmp)

#     pool.close()
#     pool.join()

#     return camera_list

    # process.start()
    # pool.append(process)

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
