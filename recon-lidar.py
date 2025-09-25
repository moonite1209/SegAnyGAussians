#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D重建和激光雷达数据处理工具
功能：
1. DMB文件可视化
2. 基于COLMAP位姿的密集点云融合
"""

import numpy as np
import cv2
import os
import json
import struct
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
from pathlib import Path
import argparse
from collections import defaultdict
from scipy.spatial import KDTree
from skimage import measure
import time
from concurrent.futures import ThreadPoolExecutor
try:
    import open3d as o3d
except Exception:
    o3d = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DMBProcessor:
    """DMB文件处理器"""
    
    def __init__(self):
        self.depth_maps = {}
        self.confidence_maps = {}
        self.camera_params = {}
    
    def read_dmb_file(self, file_path, is_confidence=False):
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
                
                print(f"DMB文件信息: type={type_val}, 高度={h}, 宽度={w}, 通道数={nb}")
                
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
    
    def visualize_depth_map(self, depth_array, title="深度图", save_path=None, colormap='plasma'):
        """
        可视化深度图
        
        参数:
            depth_array: 深度数据数组
            title: 图像标题
            save_path: 保存路径（可选）
            colormap: 颜色映射
        """
        plt.figure(figsize=(10, 8))
        
        # 过滤无效值
        valid_depth = depth_array.copy()
        valid_depth[valid_depth <= 0] = np.nan
        
        plt.imshow(valid_depth, cmap=colormap)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"深度图可视化已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_confidence_map(self, confidence_array, title="置信度图", save_path=None):
        """
        可视化置信度图（离散值：0, 1, 2）
        
        参数:
            confidence_array: 置信度数据数组
            title: 图像标题
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 8))
        
        # 使用灰度图可视化置信度
        plt.imshow(confidence_array, cmap='gray', vmin=0, vmax=2)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"置信度图可视化已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    def process_data_folder(self, data_folder, output_folder=None):
        """
        处理整个数据文件夹中的DMB文件，确保深度图和其对应的置信度图被一起处理。
        
        参数:
            data_folder: 数据文件夹路径
            output_folder: 输出文件夹路径（可选）
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # 查找所有深度文件作为处理的起点
        depth_files = sorted(glob.glob(os.path.join(data_folder, "*_smoothDepth.dmb")))
        
        print(f"找到 {len(depth_files)} 个深度文件")
        
        # 处理前5个找到的深度文件及其对应的置信度文件
        for i, depth_file in enumerate(depth_files[:5]):
            base_name_with_ext = os.path.basename(depth_file)
            base_name = base_name_with_ext.replace('_smoothDepth.dmb', '')
            
            # 查找对应的置信度文件
            conf_file = os.path.join(data_folder, f"{base_name}_confidence.dmb")
            
            print(f"处理文件对 {i+1}/{min(5, len(depth_files))}: {base_name}")
            
            # 处理深度图
            print(f"  -> 深度图: {os.path.basename(depth_file)}")
            depth_data = self.read_dmb_file(depth_file, is_confidence=False)
            if depth_data is not None:
                save_path = None
                if output_folder:
                    save_path = os.path.join(output_folder, f"{base_name}_smoothDepth_visualization.png")
                self.visualize_depth_map(depth_data, f"Depth Map - {base_name}", save_path)
            
            # 处理置信度图
            if os.path.exists(conf_file):
                print(f"  -> 置信度图: {os.path.basename(conf_file)}")
                conf_data = self.read_dmb_file(conf_file, is_confidence=True)
                if conf_data is not None:
                    save_path = None
                    if output_folder:
                        save_path = os.path.join(output_folder, f"{base_name}_confidence_visualization.png")
                    self.visualize_confidence_map(conf_data, f"Confidence Map - {base_name}", save_path)
            else:
                print(f"  -> 未找到对应的置信度文件: {os.path.basename(conf_file)}")


class PointCloudFusion:
    """基于COLMAP位姿的密集点云融合器 - 支持多种融合方式"""
    
    def __init__(self, transpose_extrinsics: bool = False, fusion_method: str = "simple", colmap_path: str = None, normals_from_depth: bool = False):
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        # 某些来源(如iOS/ARKit/OpenGL)的矩阵可能为列主序，需转置
        self.transpose_extrinsics = transpose_extrinsics
        # 融合方法：simple, consistency, tsdf, probabilistic
        self.fusion_method = fusion_method
        # COLMAP模型路径（包含 cameras.txt/images.txt 或 bin）
        self.colmap_path = colmap_path
        # 是否由深度图计算法线
        self.normals_from_depth = normals_from_depth
        # 是否保存中间结果
        self.save_intermediates = False
    
    def load_camera_params(self, json_file):
        """
        从JSON文件加载相机参数
        
        参数:
            json_file: JSON相机参数文件路径
            
        返回:
            相机内参矩阵和外参矩阵
        """
        try:
            with open(json_file, 'r') as f:
                params = json.load(f)
            
            # 提取内参矩阵
            intrinsics = np.array(params['cameraIntrinsics'])
            
            # 提取外参矩阵（世界到本地变换）
            world_to_local = np.array(params['worldToLocal'])
            if self.transpose_extrinsics:
                world_to_local = world_to_local.T

            # 位姿体检
            if world_to_local.shape == (4, 4):
                R = world_to_local[:3, :3]
                t = world_to_local[:3, 3]
                ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
                det_R = np.linalg.det(R)
                bottom_row = world_to_local[3, :]
                print(f"  位姿检查: ||R^T R - I||={ortho_err:.2e}, det(R)={det_R:.4f}, bottom={bottom_row}")
            else:
                print("  警告: worldToLocal形状非4x4")
            
            return intrinsics, world_to_local
            
        except Exception as e:
            print(f"加载相机参数失败 {json_file}: {e}")
            return None, None

    def scale_intrinsics(self, intrinsics: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
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

    def _compute_normals_from_depth(self, depth: np.ndarray, K: np.ndarray, pose_world_to_cam: np.ndarray) -> np.ndarray:
        """由深度图计算法线（相机系叉乘），并旋转到世界系。
        返回形状 (H, W, 3)，无效像素为0。
        """
        h, w = depth.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        # 像素网格
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        z = depth.astype(np.float32)
        valid = z > 0
        # 相机坐标点云（所有像素）
        x = (uu - cx) * z / (fx + 1e-8)
        y = (vv - cy) * z / (fy + 1e-8)
        # 邻域向量（u方向、v方向），内部区域计算
        # u方向: P(u+1,v) - P(u,v)
        tx = (x[:, 1:] - x[:, :-1])
        ty = (y[:, 1:] - y[:, :-1])
        tz = (z[:, 1:] - z[:, :-1])
        # v方向: P(u,v+1) - P(u,v)
        sx = (x[1:, :] - x[:-1, :])
        sy = (y[1:, :] - y[:-1, :])
        sz = (z[1:, :] - z[:-1, :])
        # 为了对齐尺寸，取内部 (h-1,w-1) 交叉区域计算法线
        t = np.stack([tx[:-1, :], ty[:-1, :], tz[:-1, :]], axis=-1)  # (h-1,w-1,3)
        s = np.stack([sx[:, :-1], sy[:, :-1], sz[:, :-1]], axis=-1)  # (h-1,w-1,3)
        n_cam = np.cross(t, s)  # (h-1,w-1,3)
        # 归一化
        n_norm = np.linalg.norm(n_cam, axis=-1, keepdims=True) + 1e-8
        n_cam = n_cam / n_norm
        # 朝向修正：使法线朝向相机（相机在原点，点在 +Z 方向），若 dot(n, P)>0 则翻转
        pcx = x[:-1, :-1]
        pcy = y[:-1, :-1]
        pcz = z[:-1, :-1]
        dot_np = n_cam[:, :, 0] * pcx + n_cam[:, :, 1] * pcy + n_cam[:, :, 2] * pcz
        flip_mask = dot_np > 0
        if np.any(flip_mask):
            n_cam[flip_mask] = -n_cam[flip_mask]
        # 无效像素（任一参与像素无效）置零
        valid_core = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
        n_cam[~valid_core] = 0
        # 旋转到世界系
        R_cw = np.linalg.inv(pose_world_to_cam)[:3, :3].astype(np.float32)
        n_cam_flat = n_cam.reshape(-1, 3).T  # (3, N)
        n_world_flat = (R_cw @ n_cam_flat).T
        n_world = np.zeros((h, w, 3), dtype=np.float32)
        n_world[:-1, :-1, :] = n_world_flat.reshape(h - 1, w - 1, 3)
        # 最后一行/列简单复制相邻（可选）
        n_world[-1, :-1, :] = n_world[-2, :-1, :]
        n_world[:-1, -1, :] = n_world[:-1, -2, :]
        n_world[-1, -1, :] = n_world[-2, -2, :]
        return n_world

    def _save_normal_map(self, normals_field: np.ndarray, out_path: str):
        try:
            n = normals_field.copy()
            nlen = np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
            n = n / nlen
            # 映射到 [0,255]
            n_img = ((n * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            # OpenCV使用BGR
            n_img_bgr = n_img[:, :, ::-1]
            cv2.imwrite(out_path, n_img_bgr)
            print(f"保存法线贴图: {out_path}")
        except Exception as e:
            print(f"保存法线贴图失败: {e}")

    # ================= COLMAP 读取与适配 =================
    def _qvec2rotmat(self, qvec: np.ndarray) -> np.ndarray:
        """COLMAP四元数(qw,qx,qy,qz)转旋转矩阵(R)
        参考COLMAP read_model.py
        """
        qw, qx, qy, qz = qvec
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
        ], dtype=np.float64)

    def _read_colmap_text_cameras(self, cameras_txt: str):
        cams = {}
        with open(cameras_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                cam_id = int(tokens[0])
                model = tokens[1]
                width = int(tokens[2])
                height = int(tokens[3])
                params = np.array([float(x) for x in tokens[4:]], dtype=np.float64)
                cams[cam_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params,
                }
        return cams

    def _read_colmap_text_images(self, images_txt: str):
        imgs = {}
        with open(images_txt, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            image_id = int(tokens[0])
            qvec = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])], dtype=np.float64)
            tvec = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])], dtype=np.float64)
            camera_id = int(tokens[8])
            name = ' '.join(tokens[9:])
            # 下一行是2D观测，可忽略
            if i < len(lines) and lines[i] and not lines[i].startswith('#'):
                i += 1
            R = self._qvec2rotmat(qvec)
            # world_to_cam: X_cam = R * X_world + t
            world_to_cam = np.eye(4, dtype=np.float64)
            world_to_cam[:3, :3] = R
            world_to_cam[:3, 3] = tvec
            imgs[image_id] = {
                'camera_id': camera_id,
                'name': name,
                'world_to_cam': world_to_cam,
            }
        return imgs

    def _camera_to_K(self, cam: dict) -> np.ndarray:
        model = cam['model'].upper()
        p = cam['params']
        if model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV']:
            fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        elif model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            fx, fy, cx, cy = float(p[0]), float(p[0]), float(p[1]), float(p[2])
        else:
            # 兜底：用width/height居中，焦距取width的等效
            fx = fy = float(cam['width'])
            cx = float(cam['width']) * 0.5
            cy = float(cam['height']) * 0.5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K

    def _collect_view_data_colmap(self, data_folder: str, limit=None):
        """使用COLMAP位姿与内参收集视角数据"""
        if self.colmap_path is None:
            return []
        # 优先读取bin
        cameras_bin = os.path.join(self.colmap_path, 'cameras.bin')
        images_bin = os.path.join(self.colmap_path, 'images.bin')
        if os.path.exists(cameras_bin) and os.path.exists(images_bin):
            cameras, images = self._read_colmap_bin(cameras_bin, images_bin)
            if cameras is None or images is None:
                print("读取COLMAP二进制失败，回退到文本")
                cameras = None
                images = None
        else:
            cameras = None
            images = None

        # 次选读取文本
        if cameras is None or images is None:
            cameras_txt = os.path.join(self.colmap_path, 'cameras.txt')
            images_txt = os.path.join(self.colmap_path, 'images.txt')
            if not (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
                print("未找到 cameras(.bin/.txt) 或 images(.bin/.txt)，COLMAP集成被跳过")
                return []
            cameras = self._read_colmap_text_cameras(cameras_txt)
            images = self._read_colmap_text_images(images_txt)
        # 深度/置信度列表
        depth_files = glob.glob(os.path.join(data_folder, "*_smoothDepth.dmb"))
        confidence_files = glob.glob(os.path.join(data_folder, "*_confidence.dmb"))
        view_data = []
        dmb_processor = DMBProcessor()
        count = 0
        for image_id in sorted(images.keys()):
            if limit is not None and count >= limit:
                break
            img = images[image_id]
            cam = cameras.get(img['camera_id'])
            if cam is None:
                continue
            base_name = os.path.splitext(os.path.basename(img['name']))[0]
            # 匹配对应深度/置信度
            depth_file = None
            for df in depth_files:
                if base_name in df:
                    depth_file = df
                    break
            if depth_file is None:
                continue
            confidence_file = None
            for cf in confidence_files:
                if base_name in cf:
                    confidence_file = cf
                    break
            # 读取深度与置信度
            depth_map = dmb_processor.read_dmb_file(depth_file, is_confidence=False)
            if depth_map is None:
                continue
            conf_map = None
            if confidence_file:
                conf_map = dmb_processor.read_dmb_file(confidence_file, is_confidence=True)
            # K缩放至深度图尺寸
            depth_h, depth_w = depth_map.shape[:2]
            K = self._camera_to_K(cam)
            src_w, src_h = int(cam['width']), int(cam['height'])
            K_scaled = self.scale_intrinsics(K, src_w, src_h, depth_w, depth_h)
            # 外参 world_to_local == world_to_cam
            world_to_local = img['world_to_cam'].astype(np.float64)
            view_data.append((depth_map, K_scaled, world_to_local, conf_map, base_name))
            count += 1
        return view_data

    def _read_colmap_bin(self, cameras_bin_path: str, images_bin_path: str):
        """读取标准COLMAP二进制 cameras.bin 与 images.bin"""
        try:
            with open(cameras_bin_path, 'rb') as f:
                num_cams_bytes = f.read(8)
                if len(num_cams_bytes) < 8:
                    return None, None
                num_cams = struct.unpack('<Q', num_cams_bytes)[0]
                cameras = {}
                for _ in range(num_cams):
                    cam_id = struct.unpack('<i', f.read(4))[0]
                    model_id = struct.unpack('<i', f.read(4))[0]
                    width = struct.unpack('<Q', f.read(8))[0]
                    height = struct.unpack('<Q', f.read(8))[0]
                    # 参数个数
                    params_num = self._colmap_model_params_num(model_id)
                    params = list(struct.unpack('<' + 'd'*params_num, f.read(8*params_num)))
                    cameras[cam_id] = {
                        'model_id': model_id,
                        'width': int(width),
                        'height': int(height),
                        'params': np.array(params, dtype=np.float64)
                    }
            with open(images_bin_path, 'rb') as f:
                num_imgs = struct.unpack('<Q', f.read(8))[0]
                images = {}
                for _ in range(num_imgs):
                    image_id = struct.unpack('<i', f.read(4))[0]
                    qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
                    tx, ty, tz = struct.unpack('<ddd', f.read(24))
                    camera_id = struct.unpack('<i', f.read(4))[0]
                    # 以\0结束的名称
                    name_bytes = bytearray()
                    while True:
                        c = f.read(1)
                        if not c or c == b'\x00':
                            break
                        name_bytes.extend(c)
                    name = name_bytes.decode('utf-8', errors='ignore')
                    # 跳过2D点
                    npts = struct.unpack('<Q', f.read(8))[0]
                    f.seek(npts * (8 + 8 + 8), os.SEEK_CUR)  # x(double)+y(double)+point3d_id(int64)
                    # 旋转
                    R = self._qvec2rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))
                    world_to_cam = np.eye(4, dtype=np.float64)
                    world_to_cam[:3, :3] = R
                    world_to_cam[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
                    images[image_id] = {
                        'camera_id': camera_id,
                        'name': name,
                        'world_to_cam': world_to_cam,
                    }
            # 将camera模型补充model字符串
            for cam_id, cam in cameras.items():
                cam['model'] = self._colmap_model_name(cam['model_id'])
            return cameras, images
        except Exception as e:
            print(f"读取COLMAP二进制失败: {e}")
            return None, None

    def _colmap_model_params_num(self, model_id: int) -> int:
        mapping = {
            0: 3,   # SIMPLE_PINHOLE: f, cx, cy
            1: 4,   # PINHOLE: fx, fy, cx, cy
            2: 4,   # SIMPLE_RADIAL: f, cx, cy, k1
            3: 5,   # RADIAL: f, cx, cy, k1, k2
            4: 8,   # OPENCV
            5: 8,   # OPENCV_FISHEYE
            6: 12,  # FULL_OPENCV
            7: 5,   # FOV
            8: 12,  # THIN_PRISM_FISHEYE
            9: 8,   # DOUBLE_SPHERE (fx, fy, cx, cy, ...)
            10: 4,  # SIMPLE_RADIAL_FISHEYE
            11: 5,  # RADIAL_FISHEYE
        }
        return mapping.get(int(model_id), 4)

    def _colmap_model_name(self, model_id: int) -> str:
        mapping = {
            0: 'SIMPLE_PINHOLE',
            1: 'PINHOLE',
            2: 'SIMPLE_RADIAL',
            3: 'RADIAL',
            4: 'OPENCV',
            5: 'OPENCV_FISHEYE',
            6: 'FULL_OPENCV',
            7: 'FOV',
            8: 'THIN_PRISM_FISHEYE',
            9: 'DOUBLE_SPHERE',
            10: 'SIMPLE_RADIAL_FISHEYE',
            11: 'RADIAL_FISHEYE',
        }
        return mapping.get(int(model_id), 'PINHOLE')

    def _read_colmap_points3d_bin(self, points_bin_path: str):
        """读取COLMAP points3D.bin文件。"""
        try:
            with open(points_bin_path, 'rb') as f:
                num_points = struct.unpack('<Q', f.read(8))[0]
                points_xyz_list = []
                points_rgb_list = []
                for _ in range(num_points):
                    point_id = struct.unpack('<Q', f.read(8))[0]
                    xyz = struct.unpack('<ddd', f.read(24))
                    rgb = struct.unpack('<BBB', f.read(3))
                    error = struct.unpack('<d', f.read(8))[0]
                    track_len = struct.unpack('<Q', f.read(8))[0]
                    f.seek(track_len * 8, os.SEEK_CUR)  # image_id(uint32)+point2d_idx(uint32)
                    points_xyz_list.append(xyz)
                    points_rgb_list.append(rgb)
            
            points_xyz = np.array(points_xyz_list, dtype=np.float64)
            points_rgb = np.array(points_rgb_list, dtype=np.uint8)
            return points_xyz, points_rgb
        except Exception as e:
            print(f"读取 points3D.bin 失败: {e}")
            return None, None
            
    def _read_colmap_points3d_txt(self, points_txt_path: str):
        """读取COLMAP points3D.txt文件。"""
        points_xyz_list = []
        points_rgb_list = []
        try:
            with open(points_txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 7:
                        points_xyz_list.append([float(p) for p in parts[1:4]])
                        points_rgb_list.append([int(p) for p in parts[4:7]])
            
            points_xyz = np.array(points_xyz_list, dtype=np.float64)
            points_rgb = np.array(points_rgb_list, dtype=np.uint8)
            return points_xyz, points_rgb
        except Exception as e:
            print(f"读取 points3D.txt 失败: {e}")
            return None, None

    def _load_colmap_sparse_points(self):
        """从COLMAP模型加载稀疏点云。"""
        if self.colmap_path is None:
            return None, None
        
        points_bin = os.path.join(self.colmap_path, 'points3D.bin')
        points_txt = os.path.join(self.colmap_path, 'points3D.txt')
        
        points_xyz, points_rgb = None, None
        
        if os.path.exists(points_bin):
            print("  正在从 points3D.bin 加载COLMAP稀疏点云...")
            points_xyz, points_rgb = self._read_colmap_points3d_bin(points_bin)
        elif os.path.exists(points_txt):
            print("  正在从 points3D.txt 加载COLMAP稀疏点云...")
            points_xyz, points_rgb = self._read_colmap_points3d_txt(points_txt)
            
        if points_xyz is not None:
            print(f"  已加载 {len(points_xyz)} 个来自COLMAP的稀疏点。")
            
        return points_xyz, points_rgb
    
    def depth_to_pointcloud(self, depth_map, intrinsics, extrinsics, confidence_map=None, rgb_image=None, min_depth=0.01, max_depth=100.0):
        """
        将深度图转换为点云
        
        参数:
            depth_map: 深度图
            intrinsics: 相机内参矩阵
            extrinsics: 相机外参矩阵
            confidence_map: 置信度图（可选）
            rgb_image: RGB图像用于着色（可选）
            min_depth: 最小深度阈值
            max_depth: 最大深度阈值
            
        返回:
            points: 3D点云坐标数组 (N, 3)
            colors: RGB颜色数组 (N, 3)，如果提供了rgb_image
        """
        h, w = depth_map.shape
        
        # 分析深度值分布
        valid_depths = depth_map[depth_map > 0]
        if len(valid_depths) > 0:
            print(f"    深度值范围: {np.min(valid_depths):.3f} - {np.max(valid_depths):.3f}")
            print(f"    平均深度: {np.mean(valid_depths):.3f}")
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 有效深度掩码
        valid_mask = (depth_map > min_depth) & (depth_map < max_depth) & (depth_map > 0)
        
        # 如果有置信度图，使用置信度阈值
        if confidence_map is not None:
            confidence_threshold = 0  # 只使用置信度级别 > 0 的像素 (0=低, 1=中, 2=高)
            valid_mask = valid_mask & (confidence_map > confidence_threshold)
            print(f"    使用置信度阈值: > {confidence_threshold} (接受级别1和2)")
        
        # 提取有效像素
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        
        print(f"    有效像素数: {len(depth_valid)} / {h*w} ({len(depth_valid)/(h*w)*100:.1f}%)")
        
        if len(depth_valid) == 0:
            return np.array([]).reshape(0, 3)
        
        # 像素坐标转相机坐标
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        # 构建相机坐标系下的点云
        points_cam = np.vstack([x_cam, y_cam, z_cam, np.ones(len(x_cam))])
        
        # 转换到世界坐标系
        # 注意：如果worldToLocal是从世界到本地的变换，我们需要其逆变换
        local_to_world = np.linalg.inv(extrinsics)
        points_world = local_to_world @ points_cam
        
        # 提取RGB颜色（如果提供了RGB图像）
        colors = None
        if rgb_image is not None:
            # 确保RGB图像是正确的格式
            if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                # 从RGB图像中采样颜色，需要将深度图坐标映射到RGB图像坐标
                rgb_h, rgb_w = rgb_image.shape[:2]
                
                # 如果深度图和RGB图像尺寸不同，需要缩放坐标
                scale_u = rgb_w / w
                scale_v = rgb_h / h
                
                # 缩放像素坐标到RGB图像
                u_rgb = (u_valid * scale_u).astype(int)
                v_rgb = (v_valid * scale_v).astype(int)
                
                # 确保坐标在范围内
                u_rgb = np.clip(u_rgb, 0, rgb_w - 1)
                v_rgb = np.clip(v_rgb, 0, rgb_h - 1)
                
                # 提取RGB值
                colors = rgb_image[v_rgb, u_rgb]  # (N, 3) BGR格式
                # 转换为RGB格式（OpenCV默认是BGR）
                colors = colors[:, ::-1]  # BGR -> RGB
        
        if colors is not None:
            return points_world[:3].T, colors  # 返回Nx3的点云和Nx3的颜色
        else:
            return points_world[:3].T  # 返回Nx3的点云
    
    def fuse_point_clouds(self, data_folder, output_file="fused_pointcloud.ply", limit=None):
        """
        融合多个深度图生成密集点云
        
        参数:
            data_folder: 数据文件夹路径
            output_file: 输出点云文件路径
        """
        print("开始点云融合...")
        
        # 若指定COLMAP路径，优先使用COLMAP的位姿与相机
        use_colmap = False
        if self.colmap_path is not None:
            cams_txt = os.path.join(self.colmap_path, 'cameras.txt')
            imgs_txt = os.path.join(self.colmap_path, 'images.txt')
            use_colmap = os.path.exists(cams_txt) and os.path.exists(imgs_txt)

        all_points = []
        all_colors = []  # 存储颜色信息

        if use_colmap:
            print("使用COLMAP位姿进行简单融合")
            # 加载并添加COLMAP稀疏点云
            sparse_points, sparse_colors = self._load_colmap_sparse_points()
            if sparse_points is not None:
                all_points.append(sparse_points)
                if sparse_colors is not None:
                    all_colors.append(sparse_colors)

            view_data = self._collect_view_data_colmap(data_folder, limit=limit)
            for i, (depth_map, K, extrinsics, conf_map, base_name) in enumerate(view_data):
                print(f"处理 {i+1}/{len(view_data)}: {base_name}")
                inter_dir = os.path.join(os.path.dirname(output_file), 'intermediates')
                if self.save_intermediates:
                    os.makedirs(inter_dir, exist_ok=True)
                # 加载RGB
                rgb_image = None
                rgb_path = os.path.join(data_folder, f"{base_name}.jpg")
                if os.path.exists(rgb_path):
                    rgb_image = cv2.imread(rgb_path)
                    if rgb_image is not None:
                        print(f"  加载RGB图像: {rgb_image.shape}")
                # 点云
                result = self.depth_to_pointcloud(depth_map, K, extrinsics, conf_map, rgb_image)
                # 保存法线贴图（如开启从深度求法线）
                if self.save_intermediates and self.normals_from_depth:
                    normals_field = self._compute_normals_from_depth(depth_map, K, extrinsics)
                    self._save_normal_map(normals_field, os.path.join(inter_dir, f"{base_name}_normals.png"))
                if isinstance(result, tuple):
                    pts, cols = result
                    if len(pts) > 0:
                        all_points.append(pts)
                        all_colors.append(cols)
                        print(f"  生成 {len(pts)} 个彩色点")
                        if self.save_intermediates:
                            self.save_ply(pts, os.path.join(inter_dir, f"{base_name}.ply"), colors=cols)
                else:
                    pts = result
                    if len(pts) > 0:
                        all_points.append(pts)
                        print(f"  生成 {len(pts)} 个点")
                        if self.save_intermediates:
                            self.save_ply(pts, os.path.join(inter_dir, f"{base_name}.ply"))
        else:
            # 回退使用JSON位姿
            json_files = glob.glob(os.path.join(data_folder, "*.json"))
            depth_files = glob.glob(os.path.join(data_folder, "*_smoothDepth.dmb"))
            confidence_files = glob.glob(os.path.join(data_folder, "*_confidence.dmb"))
            print(f"找到 {len(json_files)} 个JSON文件")
            print(f"找到 {len(depth_files)} 个深度文件")
            print(f"找到 {len(confidence_files)} 个置信度文件")
            dmb_processor = DMBProcessor()
            json_iter = json_files
            total = len(json_files)
            if isinstance(limit, int) and limit > 0:
                json_iter = json_files[:limit]
                total = min(limit, len(json_files))
            for i, json_file in enumerate(json_iter):
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                depth_file = None
                for df in depth_files:
                    if base_name in df:
                        depth_file = df
                        break
                if depth_file is None:
                    print(f"未找到对应的深度文件: {base_name}")
                    continue
                confidence_file = None
                for cf in confidence_files:
                    if base_name in cf:
                        confidence_file = cf
                        break
                print(f"处理 {i+1}/{total}: {base_name}")
                intrinsics, extrinsics = self.load_camera_params(json_file)
                if intrinsics is None:
                    continue
                depth_map = dmb_processor.read_dmb_file(depth_file, is_confidence=False)
                if depth_map is None:
                    continue
                confidence_map = None
                if confidence_file:
                    confidence_map = dmb_processor.read_dmb_file(confidence_file, is_confidence=True)
                depth_h, depth_w = depth_map.shape[:2]
                inter_dir = os.path.join(os.path.dirname(output_file), 'intermediates')
                if self.save_intermediates:
                    os.makedirs(inter_dir, exist_ok=True)
                rgb_path = os.path.join(data_folder, f"{base_name}.jpg")
                if os.path.exists(rgb_path):
                    rgb_img = cv2.imread(rgb_path)
                    if rgb_img is not None:
                        src_h, src_w = rgb_img.shape[:2]
                    else:
                        src_w = int(round(float(intrinsics[0, 2]) * 2.0))
                        src_h = int(round(float(intrinsics[1, 2]) * 2.0))
                else:
                    src_w = int(round(float(intrinsics[0, 2]) * 2.0))
                    src_h = int(round(float(intrinsics[1, 2]) * 2.0))
                K_scaled = self.scale_intrinsics(intrinsics, src_w, src_h, depth_w, depth_h)
                print(f"  K原始: fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}, cx={intrinsics[0,2]:.2f}, cy={intrinsics[1,2]:.2f}")
                print(f"  K缩放: fx={K_scaled[0,0]:.2f}, fy={K_scaled[1,1]:.2f}, cx={K_scaled[0,2]:.2f}, cy={K_scaled[1,2]:.2f}")
                rgb_image = None
                if os.path.exists(rgb_path):
                    rgb_image = cv2.imread(rgb_path)
                    if rgb_image is not None:
                        print(f"  加载RGB图像: {rgb_image.shape}")
                result = self.depth_to_pointcloud(depth_map, K_scaled, extrinsics, confidence_map, rgb_image)
                if self.save_intermediates and self.normals_from_depth:
                    normals_field = self._compute_normals_from_depth(depth_map, K_scaled, extrinsics)
                    self._save_normal_map(normals_field, os.path.join(inter_dir, f"{base_name}_normals.png"))
                if isinstance(result, tuple):
                    points, colors = result
                    if len(points) > 0:
                        all_points.append(points)
                        all_colors.append(colors)
                        print(f"  生成 {len(points)} 个彩色点")
                        if self.save_intermediates:
                            self.save_ply(points, os.path.join(inter_dir, f"{base_name}.ply"), colors=colors)
                else:
                    points = result
                    if len(points) > 0:
                        all_points.append(points)
                        print(f"  生成 {len(points)} 个点")
                        if self.save_intermediates:
                            self.save_ply(points, os.path.join(inter_dir, f"{base_name}.ply"))
        
        if len(all_points) == 0:
            print("没有生成任何点云数据")
            return
        
        # 合并所有点云
        fused_points = np.vstack(all_points)
        print(f"融合后总点数: {len(fused_points)}")
        
        # 合并颜色信息（如果有）
        fused_colors = None
        if len(all_colors) > 0:
            fused_colors = np.vstack(all_colors)
            print(f"合并颜色信息: {len(fused_colors)} 个点")
        
        # 保存为PLY格式
        self.save_ply(fused_points, output_file, colors=fused_colors)
    
    def save_ply(self, points, filename, normals=None, colors=None):
        """
        保存点云为PLY格式
        
        参数:
            points: Nx3的点云数组
            filename: 输出文件名
            normals: Nx3的法向量数组（可选）
            colors: Nx3的颜色数组（可选）
        """
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z"""
        
        if normals is not None:
            header += """
property float nx
property float ny
property float nz"""
        
        if colors is not None:
            header += """
property uchar red
property uchar green
property uchar blue"""
        
        header += "\nend_header\n"
        
        with open(filename, 'w') as f:
            f.write(header)
            for i, point in enumerate(points):
                line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}"
                if normals is not None:
                    line += f" {normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f}"
                if colors is not None:
                    line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
                line += "\n"
                f.write(line)
        
        print(f"点云已保存到: {filename}")
    
    # === 1) 点云式融合：几何/可见性一致性 ===
    def geometric_consistency_fusion(self, data_folder, output_file, voxel_size=0.02, 
                                   min_views=2, depth_threshold=0.05, pixel_threshold=2.0, limit=None,
                                   workers: int = 0):
        """
        基于几何一致性的点云融合（向量化优化版本）
        
        参数:
            data_folder: 数据文件夹
            output_file: 输出文件
            voxel_size: 体素大小用于聚合
            min_views: 最小一致视角数
            depth_threshold: 深度一致性阈值
            pixel_threshold: 像素投影误差阈值
            limit: 限制处理的视角数量
        """
        print("=== 几何一致性点云融合（向量化优化） ===")
        
        # 收集所有数据（优先COLMAP）
        view_data = []
        if self.colmap_path is not None and os.path.exists(os.path.join(self.colmap_path, 'cameras.txt')):
            view_data = self._collect_view_data_colmap(data_folder, limit=limit)
        if len(view_data) == 0:
            view_data = self._collect_view_data(data_folder, limit=limit)
        if len(view_data) == 0:
            print("未找到有效数据")
            return
        t_total_start = time.time()

        # 体素网格用于空间聚合
        voxel_grid = defaultdict(list)  # voxel_key -> list of (point, normal, view_id, confidence)
        
        # 对每个视角生成点云并进行一致性检查
        for view_id, (depth_map, K, pose, conf_map, base_name) in enumerate(view_data):
            print(f"处理视角 {view_id+1}/{len(view_data)}")
            t_view_start = time.time()
            
            h, w = depth_map.shape
            
            # 使用向量化操作生成所有3D点
            # 创建像素坐标网格
            u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
            
            # 有效深度掩码
            valid_mask = (depth_map > 0) & (depth_map <= 100.0)  # Fixed: use same max depth as Rust
            if conf_map is not None:
                valid_mask = valid_mask & (conf_map > 0)
            
            # 提取有效像素
            u_valid = u_grid[valid_mask]
            v_valid = v_grid[valid_mask]
            d_valid = depth_map[valid_mask]
            conf_valid = np.ones_like(d_valid)
            if conf_map is not None:
                conf_valid = conf_map[valid_mask] / 2.0
            
            if len(d_valid) == 0:
                continue
            
            # 向量化反投影到相机坐标
            x_cam = (u_valid - K[0,2]) * d_valid / K[0,0]
            y_cam = (v_valid - K[1,2]) * d_valid / K[1,1]
            z_cam = d_valid
            
            # 构建齐次坐标
            points_cam_homo = np.vstack([x_cam, y_cam, z_cam, np.ones(len(x_cam))])
            
            # 向量化转换到世界坐标
            local_to_world = np.linalg.inv(pose)
            points_world_homo = local_to_world @ points_cam_homo
            points_world = points_world_homo[:3].T  # (N, 3)
            
            # 法线计算（可选：由深度图估计）
            if self.normals_from_depth:
                normals_field = self._compute_normals_from_depth(depth_map, K, pose)
                normals = normals_field[v_valid, u_valid]
                nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
                normals = normals / nlen
            else:
                cam_center = local_to_world @ np.array([0, 0, 0, 1])
                normals = cam_center[:3][None, :] - points_world  # 广播
                norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / (norm_lengths + 1e-8)
            
            # 向量化一致性检查
            valid_points = []
            valid_normals = []
            valid_confidences = []
            
            # 对其他视角进行批量投影检查
            consistent_counts = np.ones(len(points_world), dtype=np.int32)  # 当前视角总是一致的

            # 预先构建一次齐次坐标，避免在并行中重复创建
            points_world_homo_check = np.column_stack([points_world, np.ones(len(points_world), dtype=points_world.dtype)])

            t_consistency_start = time.time()

            def compute_consistency(other_idx: int) -> np.ndarray:
                other_depth, other_K, other_pose, _, _ = view_data[other_idx]
                points_other_cam = (other_pose @ points_world_homo_check.T).T
                behind_mask = points_other_cam[:, 2] > 0
                if not np.any(behind_mask):
                    return np.zeros(len(points_world), dtype=np.uint8)
                u_proj = (other_K[0,0] * points_other_cam[:, 0] / points_other_cam[:, 2] + other_K[0,2]).astype(int)
                v_proj = (other_K[1,1] * points_other_cam[:, 1] / points_other_cam[:, 2] + other_K[1,2]).astype(int)
                in_bounds_mask = (u_proj >= 0) & (u_proj < other_depth.shape[1]) & \
                                 (v_proj >= 0) & (v_proj < other_depth.shape[0]) & behind_mask
                if not np.any(in_bounds_mask):
                    return np.zeros(len(points_world), dtype=np.uint8)
                expected_depths = points_other_cam[:, 2]
                observed_depths = np.zeros_like(expected_depths)
                valid_indices = np.where(in_bounds_mask)[0]
                observed_depths[valid_indices] = other_depth[v_proj[valid_indices], u_proj[valid_indices]]
                depth_diff = np.abs(expected_depths - observed_depths)
                consistent_mask = in_bounds_mask & (observed_depths > 0) & (depth_diff < depth_threshold)
                return consistent_mask.astype(np.uint8)

            other_indices = [idx for idx in range(len(view_data)) if idx != view_id]

            if isinstance(workers, int) and workers > 1 and len(other_indices) > 0:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for mask in executor.map(compute_consistency, other_indices):
                        if mask is not None and len(mask) == len(consistent_counts):
                            consistent_counts += mask.astype(np.int32)
            else:
                for other_id in other_indices:
                    mask = compute_consistency(other_id)
                    if mask is not None and len(mask) == len(consistent_counts):
                        consistent_counts += mask.astype(np.int32)
            
            # 筛选一致性足够的点
            final_mask = consistent_counts >= min_views
            consistency_elapsed = time.time() - t_consistency_start
            n_pass = int(np.count_nonzero(final_mask))
            if np.any(final_mask):
                final_points = points_world[final_mask]
                final_normals = normals[final_mask]
                final_confidences = conf_valid[final_mask]
                
                # 加载RGB图像用于着色
                json_files = self._get_json_files(data_folder, len(view_data))
                base_name = os.path.splitext(os.path.basename(json_files[view_id]))[0]
                rgb_path = os.path.join(data_folder, f"{base_name}.jpg")
                rgb_colors = None
                
                if os.path.exists(rgb_path):
                    rgb_image = cv2.imread(rgb_path)
                    if rgb_image is not None and len(final_points) > 0:
                        rgb_h, rgb_w = rgb_image.shape[:2]
                        # 缩放像素坐标到RGB图像
                        scale_u = rgb_w / w
                        scale_v = rgb_h / h
                        u_rgb = (u_valid[final_mask] * scale_u).astype(int)
                        v_rgb = (v_valid[final_mask] * scale_v).astype(int)
                        # 确保坐标在范围内
                        u_rgb = np.clip(u_rgb, 0, rgb_w - 1)
                        v_rgb = np.clip(v_rgb, 0, rgb_h - 1)
                        # 提取RGB值
                        rgb_colors = rgb_image[v_rgb, u_rgb][:, ::-1]  # BGR -> RGB
                
                # 向量化体素分配
                voxel_keys = (final_points / voxel_size).astype(int)
                for i, (point, normal, conf) in enumerate(zip(final_points, final_normals, final_confidences)):
                    voxel_key = tuple(voxel_keys[i])
                    color = rgb_colors[i] if rgb_colors is not None else [128, 128, 255]
                    voxel_grid[voxel_key].append((point, normal, view_id, conf, color))
                # 保存中间结果
                if self.save_intermediates and len(final_points) > 0:
                    inter_dir = os.path.join(os.path.dirname(output_file), 'intermediates')
                    os.makedirs(inter_dir, exist_ok=True)
                    self.save_ply(final_points, os.path.join(inter_dir, f"{base_name}_consistency_points.ply"), final_normals, np.asarray(rgb_colors) if rgb_colors is not None else None)
            view_elapsed = time.time() - t_view_start
        
        # 聚合体素内的点（矢量化 + 可选多线程）
        t_agg_start = time.time()
        final_points = []
        final_normals = []
        final_colors = []

        voxel_items = list(voxel_grid.items())

        def aggregate_voxel(item):
            _, point_list = item
            if len(point_list) < min_views:
                return None
            # 提取为数组
            try:
                confs = np.asarray([conf for _, _, _, conf, _ in point_list], dtype=np.float32)
                total_weight = float(np.sum(confs))
                if total_weight <= 0.0:
                    return None
                weights = confs / total_weight
                pts_arr = np.asarray([p for p, _, _, _, _ in point_list], dtype=np.float32)
                nrms_arr = np.asarray([n for _, n, _, _, _ in point_list], dtype=np.float32)
                cols_arr = np.asarray([c for _, _, _, _, c in point_list], dtype=np.float32)
                # 加权求和
                weighted_point = (weights[:, None] * pts_arr).sum(axis=0)
                weighted_normal = (weights[:, None] * nrms_arr).sum(axis=0)
                nlen = float(np.linalg.norm(weighted_normal)) + 1e-8
                weighted_normal = weighted_normal / nlen
                weighted_color = (weights[:, None] * cols_arr).sum(axis=0)
                return weighted_point, weighted_normal, weighted_color.astype(int)
            except Exception:
                return None

        if isinstance(workers, int) and workers > 1 and len(voxel_items) > 0:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for res in executor.map(aggregate_voxel, voxel_items):
                    if res is None:
                        continue
                    p, n, c = res
                    final_points.append(p)
                    final_normals.append(n)
                    final_colors.append(c)
        else:
            for item in voxel_items:
                res = aggregate_voxel(item)
                if res is None:
                    continue
                p, n, c = res
                final_points.append(p)
                final_normals.append(n)
                final_colors.append(c)
        agg_elapsed = time.time() - t_agg_start
        print(f"体素聚合耗时: {agg_elapsed:.2f}s，体素数: {len(voxel_items)}，输出点数: {len(final_points)}")
        
        if len(final_points) > 0:
            final_points = np.array(final_points)
            final_normals = np.array(final_normals)
            final_colors = np.array(final_colors)
            
            print(f"几何一致性融合完成：{len(final_points)} 个点")
            self.save_ply(final_points, output_file, final_normals, final_colors)
        else:
            print("几何一致性融合失败：没有生成有效点")
        total_elapsed = time.time() - t_total_start
        print(f"几何一致性融合总耗时: {total_elapsed:.2f}s")
    
    # === 2) TSDF体素融合 ===
    def tsdf_fusion(self, data_folder, output_file, voxel_size=0.02, truncation_distance=0.05, limit=None):
        """
        TSDF (Truncated Signed Distance Function) 体素融合
        
        参数:
            data_folder: 数据文件夹
            output_file: 输出文件
            voxel_size: 体素大小
            truncation_distance: 截断距离
            limit: 限制处理的视角数量
        """
        print("=== TSDF体素融合 ===")
        
        # 收集所有数据（优先COLMAP）
        view_data = []
        if self.colmap_path is not None and os.path.exists(os.path.join(self.colmap_path, 'cameras.txt')):
            view_data = self._collect_view_data_colmap(data_folder, limit=limit)
        if len(view_data) == 0:
            view_data = self._collect_view_data(data_folder, limit=limit)
        if len(view_data) == 0:
            print("未找到有效数据")
            return
        
        # 估计场景边界（向量化优化）
        all_points = []
        for depth_map, K, pose, conf_map, _ in view_data:
            # 快速采样一些点来估计边界
            h, w = depth_map.shape
            
            # 向量化采样
            v_sample = np.arange(0, h, 10)
            u_sample = np.arange(0, w, 10)
            u_grid, v_grid = np.meshgrid(u_sample, v_sample)
            u_flat, v_flat = u_grid.flatten(), v_grid.flatten()
            
            # 提取采样深度
            d_sample = depth_map[v_flat, u_flat]
            valid_mask = (d_sample > 0.1) & (d_sample < 10.0)
            
            if np.any(valid_mask):
                u_valid = u_flat[valid_mask]
                v_valid = v_flat[valid_mask]
                d_valid = d_sample[valid_mask]
                
                # 向量化反投影
                x_cam = (u_valid - K[0,2]) * d_valid / K[0,0]
                y_cam = (v_valid - K[1,2]) * d_valid / K[1,1]
                
                # 构建齐次坐标并转换
                points_cam_homo = np.vstack([x_cam, y_cam, d_valid, np.ones(len(x_cam))])
                local_to_world = np.linalg.inv(pose)
                points_world_homo = local_to_world @ points_cam_homo
                all_points.extend(points_world_homo[:3].T)
        
        if len(all_points) == 0:
            print("无法估计场景边界")
            return
        
        all_points = np.array(all_points)
        min_bound = np.min(all_points, axis=0) - 0.5
        max_bound = np.max(all_points, axis=0) + 0.5
        
        print(f"场景边界: {min_bound} - {max_bound}")
        
        # 创建体素网格
        grid_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1
        tsdf_values = np.zeros(grid_size, dtype=np.float32)
        weights = np.zeros(grid_size, dtype=np.float32)
        
        print(f"TSDF网格大小: {grid_size}")
        
        # 对每个视角更新TSDF（向量化优化）
        for view_id, (depth_map, K, pose, conf_map, _) in enumerate(view_data):
            print(f"TSDF更新视角 {view_id+1}/{len(view_data)}")
            
            h, w = depth_map.shape
            
            # 向量化生成所有体素中心
            xi_coords, yi_coords, zi_coords = np.meshgrid(
                np.arange(grid_size[0]),
                np.arange(grid_size[1]), 
                np.arange(grid_size[2]),
                indexing='ij'
            )
            
            # 展平为1D数组
            xi_flat = xi_coords.flatten()
            yi_flat = yi_coords.flatten() 
            zi_flat = zi_coords.flatten()
            
            # 计算所有体素中心的世界坐标
            voxel_centers = min_bound[None, :] + np.column_stack([xi_flat, yi_flat, zi_flat]) * voxel_size
            
            # 向量化投影到当前视角
            voxel_centers_homo = np.column_stack([voxel_centers, np.ones(len(voxel_centers))])
            points_cam = (pose @ voxel_centers_homo.T).T
            
            # 过滤在相机后面的点
            front_mask = points_cam[:, 2] > 0
            if not np.any(front_mask):
                continue
            
            # 投影到像素坐标
            u_proj = (K[0,0] * points_cam[:, 0] / points_cam[:, 2] + K[0,2]).astype(int)
            v_proj = (K[1,1] * points_cam[:, 1] / points_cam[:, 2] + K[1,2]).astype(int)
            
            # 检查边界
            in_bounds_mask = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h) & front_mask
            
            if not np.any(in_bounds_mask):
                continue
            
            # 获取有效投影的索引
            valid_indices = np.where(in_bounds_mask)[0]
            
            # 批量获取观测深度
            observed_depths = np.zeros(len(voxel_centers))
            observed_depths[valid_indices] = depth_map[v_proj[valid_indices], u_proj[valid_indices]]
            
            # 计算有符号距离
            expected_depths = points_cam[:, 2]
            sdf_values = observed_depths - expected_depths
            
            # 截断掩码
            truncation_mask = (np.abs(sdf_values) <= truncation_distance) & (observed_depths > 0) & in_bounds_mask
            if not np.any(truncation_mask):
                continue
                
            # 归一化TSDF值
            tsdf_vals = np.clip(sdf_values / truncation_distance, -1.0, 1.0)
            
            # 计算权重
            weight_vals = np.ones(len(voxel_centers))
            if conf_map is not None:
                conf_weights = np.ones(len(voxel_centers))
                conf_weights[valid_indices] = conf_map[v_proj[valid_indices], u_proj[valid_indices]]
                conf_weights[conf_weights > 0] = conf_weights[conf_weights > 0] / 2.0
                weight_vals = conf_weights
            
            # 批量更新TSDF
            update_indices = np.where(truncation_mask)[0]
            for idx in update_indices:
                xi, yi, zi = xi_flat[idx], yi_flat[idx], zi_flat[idx]
                
                old_weight = weights[xi, yi, zi]
                weight = weight_vals[idx]
                new_weight = min(old_weight + weight, 10.0)
                
                if new_weight > 0:
                    tsdf_values[xi, yi, zi] = (old_weight * tsdf_values[xi, yi, zi] + 
                                             weight * tsdf_vals[idx]) / new_weight
                    weights[xi, yi, zi] = new_weight
        
        # 使用Marching Cubes提取表面
        print("使用Marching Cubes提取表面...")
        try:
            vertices, faces, normals, _ = measure.marching_cubes(tsdf_values, level=0.0, 
                                                               spacing=(voxel_size, voxel_size, voxel_size))
            
            # 转换到世界坐标
            vertices += min_bound
            
            print(f"TSDF融合完成：{len(vertices)} 个顶点，{len(faces)} 个面")
            
            # 保存为PLY格式的网格
            self._save_mesh_ply(vertices, faces, normals, output_file.replace('.ply', '_mesh.ply'))
            
            # 也保存顶点作为点云
            colors = np.full((len(vertices), 3), [255, 128, 128])  # 红色
            self.save_ply(vertices, output_file, normals, colors)
            
        except Exception as e:
            print(f"Marching Cubes失败: {e}")
            print("保存TSDF体素...")
            # 提取非零体素作为点云
            valid_mask = weights > 0.1
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) > 0:
                voxel_points = []
                for i in range(len(valid_indices[0])):
                    xi, yi, zi = valid_indices[0][i], valid_indices[1][i], valid_indices[2][i]
                    voxel_center = min_bound + np.array([xi, yi, zi]) * voxel_size
                    voxel_points.append(voxel_center)
                
                voxel_points = np.array(voxel_points)
                colors = np.full((len(voxel_points), 3), [255, 128, 128])
                self.save_ply(voxel_points, output_file, None, colors)
                print(f"保存了 {len(voxel_points)} 个TSDF体素")
    
    # === 3) 概率式/贝叶斯融合 ===
    def probabilistic_fusion(self, data_folder, output_file, voxel_size=0.02, limit=None):
        """
        概率式/贝叶斯深度融合
        
        参数:
            data_folder: 数据文件夹
            output_file: 输出文件
            voxel_size: 体素大小
            limit: 限制处理的视角数量
        """
        print("=== 概率式贝叶斯融合 ===")
        
        # 收集所有数据（优先COLMAP）
        view_data = []
        if self.colmap_path is not None and os.path.exists(os.path.join(self.colmap_path, 'cameras.txt')):
            view_data = self._collect_view_data_colmap(data_folder, limit=limit)
        if len(view_data) == 0:
            view_data = self._collect_view_data(data_folder, limit=limit)
        if len(view_data) == 0:
            print("未找到有效数据")
            return
        
        # 体素网格存储深度分布
        voxel_grid = defaultdict(lambda: {'depths': [], 'weights': [], 'normals': [], 'colors': []})
        
        # 收集每个体素的深度观测（向量化优化）
        for view_id, (depth_map, K, pose, conf_map, base_name) in enumerate(view_data):
            print(f"收集深度观测 {view_id+1}/{len(view_data)}")
            
            h, w = depth_map.shape
            
            # 使用向量化操作处理所有像素
            u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
            
            # 有效深度掩码
            valid_mask = (depth_map > 0) & (depth_map <= 100.0)  # Fixed: use same max depth as Rust
            if conf_map is not None:
                valid_mask = valid_mask & (conf_map > 0)
            
            # 提取有效像素
            u_valid = u_grid[valid_mask]
            v_valid = v_grid[valid_mask]
            d_valid = depth_map[valid_mask]
            
            if len(d_valid) == 0:
                continue
            
            # 向量化深度噪声模型
            depth_stds = np.maximum(0.01, d_valid * 0.01)  # 1%的相对误差
            
            # 向量化置信度权重
            weights = np.ones_like(d_valid)
            if conf_map is not None:
                conf_valid = conf_map[valid_mask]
                weights = conf_valid / 2.0
            
            # 向量化反投影到相机坐标
            x_cam = (u_valid - K[0,2]) * d_valid / K[0,0]
            y_cam = (v_valid - K[1,2]) * d_valid / K[1,1]
            
            # 构建齐次坐标并转换到世界坐标
            points_cam_homo = np.vstack([x_cam, y_cam, d_valid, np.ones(len(x_cam))])
            local_to_world = np.linalg.inv(pose)
            points_world_homo = local_to_world @ points_cam_homo
            points_world = points_world_homo[:3].T  # (N, 3)
            
            # 法线计算（可选：由深度图估计）
            if self.normals_from_depth:
                normals_field = self._compute_normals_from_depth(depth_map, K, pose)
                normals = normals_field[v_valid, u_valid]
                nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
                normals = normals / nlen
            else:
                cam_center = local_to_world @ np.array([0, 0, 0, 1])
                normals = cam_center[:3][None, :] - points_world  # 广播
                norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / (norm_lengths + 1e-8)
            
            # 向量化体素键计算
            voxel_keys = (points_world / voxel_size).astype(int)
            
            # 向量化逆方差权重
            inv_var_weights = weights / (depth_stds**2)
            
            # 加载RGB图像用于着色
            json_files = self._get_json_files(data_folder, len(view_data))
            base_name = os.path.splitext(os.path.basename(json_files[view_id]))[0]
            rgb_path = os.path.join(data_folder, f"{base_name}.jpg")
            rgb_colors = None
            
            if os.path.exists(rgb_path):
                rgb_image = cv2.imread(rgb_path)
                if rgb_image is not None and len(d_valid) > 0:
                    rgb_h, rgb_w = rgb_image.shape[:2]
                    # 缩放像素坐标到RGB图像
                    scale_u = rgb_w / w
                    scale_v = rgb_h / h
                    u_rgb = (u_valid * scale_u).astype(int)
                    v_rgb = (v_valid * scale_v).astype(int)
                    # 确保坐标在范围内
                    u_rgb = np.clip(u_rgb, 0, rgb_w - 1)
                    v_rgb = np.clip(v_rgb, 0, rgb_h - 1)
                    # 提取RGB值
                    rgb_colors = rgb_image[v_rgb, u_rgb][:, ::-1]  # BGR -> RGB
            
            # 批量存储观测数据
            for i in range(len(d_valid)):
                voxel_key = tuple(voxel_keys[i])
                color = rgb_colors[i] if rgb_colors is not None else [128, 128, 128]
                voxel_grid[voxel_key]['depths'].append(d_valid[i])
                voxel_grid[voxel_key]['weights'].append(inv_var_weights[i])
                voxel_grid[voxel_key]['normals'].append(normals[i])
                voxel_grid[voxel_key]['colors'].append(color)
        
        # 贝叶斯融合
        final_points = []
        final_normals = []
        final_colors = []
        final_uncertainties = []
        
        for voxel_key, data in voxel_grid.items():
            depths = np.array(data['depths'])
            weights = np.array(data['weights'])
            normals = np.array(data['normals'])
            colors = np.array(data['colors'])
            
            if len(depths) < 2:  # 至少需要2个观测
                continue
            
            # 贝叶斯深度融合：加权平均
            total_weight = np.sum(weights)
            if total_weight == 0:
                continue
            
            fused_depth = np.sum(depths * weights) / total_weight
            fused_variance = 1.0 / total_weight
            fused_std = np.sqrt(fused_variance)
            
            # 法向量加权平均
            weighted_normal = np.average(normals, weights=weights, axis=0)
            weighted_normal = weighted_normal / (np.linalg.norm(weighted_normal) + 1e-8)
            
            # 颜色加权平均
            weighted_color = np.average(colors, weights=weights, axis=0)
            
            # 转换回3D坐标（取体素中心）
            voxel_center = np.array(voxel_key) * voxel_size + voxel_size * 0.5
            
            final_points.append(voxel_center)
            final_normals.append(weighted_normal)
            final_uncertainties.append(fused_std)
            
            # 使用RGB颜色，如果没有RGB则根据不确定性着色
            if len(colors) > 0 and not np.allclose(weighted_color, [128, 128, 128]):
                final_colors.append(weighted_color.astype(int))
            else:
                # 根据不确定性着色（蓝色=确定，红色=不确定）
                uncertainty_ratio = min(fused_std / 0.1, 1.0)  # 归一化到0-1
                color = [int(255 * uncertainty_ratio), 0, int(255 * (1 - uncertainty_ratio))]
                final_colors.append(color)
        
        if len(final_points) > 0:
            final_points = np.array(final_points)
            final_normals = np.array(final_normals)
            final_colors = np.array(final_colors)
            
            print(f"概率式融合完成：{len(final_points)} 个点")
            print(f"平均不确定性：{np.mean(final_uncertainties):.4f}")
            
            self.save_ply(final_points, output_file, final_normals, final_colors)
        else:
            print("概率式融合失败：没有生成有效点")
    
    def _get_json_files(self, data_folder, limit=None):
        """获取JSON文件列表"""
        json_files = glob.glob(os.path.join(data_folder, "*.json"))
        json_files.sort()
        if limit:
            json_files = json_files[:limit]
        return json_files
    
    def _collect_view_data(self, data_folder, limit=None):
        """收集所有视角的数据"""
        json_files = self._get_json_files(data_folder, limit)
        depth_files = glob.glob(os.path.join(data_folder, "*_smoothDepth.dmb"))
        confidence_files = glob.glob(os.path.join(data_folder, "*_confidence.dmb"))
        
        view_data = []
        dmb_processor = DMBProcessor()
        
        for json_file in json_files:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # 查找对应文件
            depth_file = None
            confidence_file = None
            
            for df in depth_files:
                if base_name in df:
                    depth_file = df
                    break
            
            for cf in confidence_files:
                if base_name in cf:
                    confidence_file = cf
                    break
            
            if depth_file is None:
                continue
            
            # 加载数据
            intrinsics, extrinsics = self.load_camera_params(json_file)
            if intrinsics is None:
                continue
            
            depth_map = dmb_processor.read_dmb_file(depth_file, is_confidence=False)
            if depth_map is None:
                continue
            
            confidence_map = None
            if confidence_file:
                confidence_map = dmb_processor.read_dmb_file(confidence_file, is_confidence=True)
            
            # 缩放内参
            depth_h, depth_w = depth_map.shape[:2]
            rgb_path = os.path.join(data_folder, f"{base_name}.jpg")
            if os.path.exists(rgb_path):
                rgb_img = cv2.imread(rgb_path)
                if rgb_img is not None:
                    src_h, src_w = rgb_img.shape[:2]
                else:
                    src_w = int(round(float(intrinsics[0, 2]) * 2.0))
                    src_h = int(round(float(intrinsics[1, 2]) * 2.0))
            else:
                src_w = int(round(float(intrinsics[0, 2]) * 2.0))
                src_h = int(round(float(intrinsics[1, 2]) * 2.0))
            
            K_scaled = self.scale_intrinsics(intrinsics, src_w, src_h, depth_w, depth_h)
            
            view_data.append((depth_map, K_scaled, extrinsics, confidence_map, base_name))
        
        return view_data
    
    def _save_mesh_ply(self, vertices, faces, normals, filename):
        """保存网格为PLY格式"""
        header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        
        with open(filename, 'w') as f:
            f.write(header)
            
            # 写入顶点
            for i, vertex in enumerate(vertices):
                normal = normals[i] if i < len(normals) else [0, 0, 1]
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                       f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            # 写入面
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"网格已保存到: {filename}")


class PointCloudModeler:
    """点云建模（网格重建）工具，基于 Open3D"""
    
    @staticmethod
    def reconstruct_from_ply(input_ply: str,
                             output_mesh: str,
                             method: str = 'poisson',
                             downsample_voxel: float = 0.0,
                             estimate_normals: bool = True,
                             poisson_depth: int = 9,
                             bpa_radius: float = 0.02,
                             alpha: float = 0.05,
                             simplify_faces: int = 0) -> bool:
        if o3d is None:
            print("未检测到Open3D，无法进行网格重建。请先安装：pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple")
            return False
        if not os.path.exists(input_ply):
            print(f"输入点云不存在: {input_ply}")
            return False
        try:
            print(f"加载点云: {input_ply}")
            pcd = o3d.io.read_point_cloud(input_ply)
            if len(np.asarray(pcd.points)) == 0:
                print("点云为空，建模终止")
                return False
            if downsample_voxel and downsample_voxel > 0:
                print(f"体素下采样: voxel={downsample_voxel}")
                pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
            # 估计法线
            if (not pcd.has_normals()) and estimate_normals:
                pts = np.asarray(pcd.points)
                bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
                diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
                search_radius = max(1e-3, diag * 0.01)
                max_nn = 30
                print(f"估计法线: radius={search_radius:.4f}, max_nn={max_nn}")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
                pcd.orient_normals_consistent_tangent_plane(k=20)
            # 网格重建
            method = method.lower()
            if method == 'poisson':
                print(f"Poisson重建: depth={poisson_depth}")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=int(poisson_depth), scale=1.1, linear_fit=True
                )
                # 裁剪到点云包围盒
                bbox = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)
            elif method == 'bpa':
                # 若未给定半径，基于最近邻距离估计
                if not bpa_radius or bpa_radius <= 0:
                    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                    pts = np.asarray(pcd.points)
                    sample = pts[::max(1, len(pts)//1000)]
                    dists = []
                    for p in sample:
                        _, idx, dist = pcd_tree.search_knn_vector_3d(o3d.utility.Vector3dVector([p])[0], 2)
                        if len(dist) > 1:
                            dists.append(np.sqrt(dist[1]))
                    median_nn = float(np.median(dists)) if len(dists) else 0.02
                    bpa_radius = max(1e-3, median_nn * 2.5)
                print(f"BPA重建: radius≈{bpa_radius:.5f}")
                radii = o3d.utility.DoubleVector([bpa_radius, bpa_radius*2.0, bpa_radius*4.0])
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            elif method == 'alpha':
                print(f"Alpha Shape重建: alpha={alpha}")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))
            else:
                print(f"未知建模方法: {method}")
                return False
            # 清理与简化
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            if simplify_faces and simplify_faces > 0:
                print(f"网格简化: 目标三角形数={simplify_faces}")
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(simplify_faces))
            mesh.compute_vertex_normals()
            # 保存
            ok = o3d.io.write_triangle_mesh(output_mesh, mesh)
            if ok:
                print(f"网格已保存到: {output_mesh}")
            else:
                print(f"写入网格失败: {output_mesh}")
            return ok
        except Exception as e:
            print(f"网格重建失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3D重建和激光雷达数据处理工具')
    parser.add_argument('--data_folder', '-d', type=str, default='data/data', 
                       help='数据文件夹路径')
    parser.add_argument('--output_folder', '-o', type=str, default='output', 
                       help='输出文件夹路径')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='可视化DMB文件')
    parser.add_argument('--fuse', '-f', action='store_true', 
                       help='融合点云')
    parser.add_argument('--fusion-method', type=str, default='simple',
                       choices=['simple', 'consistency', 'tsdf', 'probabilistic'],
                       help='融合方法：simple(简单), consistency(几何一致性), tsdf(TSDF体素), probabilistic(概率式)')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='体素大小（用于体素网格融合方法）')
    parser.add_argument('--min-views', type=int, default=2,
                       help='最小一致视角数（用于一致性融合）')
    parser.add_argument('--workers', type=int, default=0,
                       help='一致性融合时的线程数(>1启用多线程); 0/1为单线程')
    parser.add_argument('--limit', type=int, default=10,
                       help='限制处理的视角数量（用于调试）')
    parser.add_argument('--transpose-extrinsics', action='store_true',
                       help='将JSON中的worldToLocal矩阵转置后再使用（部分平台需要）')
    parser.add_argument('--colmap-path', type=str, default=None,
                       help='COLMAP重建输出目录路径（包含cameras.bin, images.bin, points3D.bin）')
    parser.add_argument('--normals-from-depth', action='store_true',
                       help='法线由深度图估计（邻域叉乘），用于一致性/概率融合')
    parser.add_argument('--save-intermediates', action='store_true',
                       help='保存中间结果（每视角点云、法线贴图）到输出目录下的intermediates')
    parser.add_argument('--output-ply', type=str, default=None,
                       help='自定义输出点云PLY文件名或路径（相对路径将基于output_folder）')
    # 建模相关
    parser.add_argument('--model', action='store_true',
                       help='开启融合点云后的网格建模')
    parser.add_argument('--model-method', type=str, default='poisson',
                       choices=['poisson', 'bpa', 'alpha'],
                       help='网格重建方法')
    parser.add_argument('--model-downsample-voxel', type=float, default=0.0,
                       help='建模前点云体素下采样大小，0表示不下采样')
    parser.add_argument('--model-poisson-depth', type=int, default=9,
                       help='Poisson重建深度(越大越细致，耗时更长)')
    parser.add_argument('--model-bpa-radius', type=float, default=0.02,
                       help='BPA半径(<=0将自动估计)')
    parser.add_argument('--model-alpha', type=float, default=0.05,
                       help='Alpha Shape参数(越小越紧致)')
    parser.add_argument('--model-simplify-faces', type=int, default=0,
                       help='Quadric简化目标三角形数，0表示不简化')
    
    args = parser.parse_args()
    
    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.visualize:
        print("=== DMB文件可视化 ===")
        processor = DMBProcessor()
        processor.process_data_folder(args.data_folder, args.output_folder)
    
    if args.fuse:
        print(f"\n=== 点云融合 (方法: {args.fusion_method}) ===")
        fusion = PointCloudFusion(transpose_extrinsics=args.transpose_extrinsics, 
                                fusion_method=args.fusion_method,
                                colmap_path=args.colmap_path,
                                normals_from_depth=args.normals_from_depth)
        # 附加: 是否保存中间结果
        fusion.save_intermediates = args.save_intermediates
        
        output_ply = None
        if args.fusion_method == 'simple':
            if args.output_ply:
                output_ply = args.output_ply if (os.path.isabs(args.output_ply) or os.path.dirname(args.output_ply)) else os.path.join(args.output_folder, args.output_ply)
            else:
                output_ply = os.path.join(args.output_folder, "fused_pointcloud_simple.ply")
            fusion.fuse_point_clouds(args.data_folder, output_ply, limit=args.limit)
        elif args.fusion_method == 'consistency':
            if args.output_ply:
                output_ply = args.output_ply if (os.path.isabs(args.output_ply) or os.path.dirname(args.output_ply)) else os.path.join(args.output_folder, args.output_ply)
            else:
                output_ply = os.path.join(args.output_folder, "fused_pointcloud_consistency.ply")
            fusion.geometric_consistency_fusion(args.data_folder, output_ply, 
                                              voxel_size=args.voxel_size, 
                                              min_views=args.min_views,
                                              limit=args.limit,
                                              workers=args.workers)
        elif args.fusion_method == 'tsdf':
            if args.output_ply:
                output_ply = args.output_ply if (os.path.isabs(args.output_ply) or os.path.dirname(args.output_ply)) else os.path.join(args.output_folder, args.output_ply)
            else:
                output_ply = os.path.join(args.output_folder, "fused_pointcloud_tsdf.ply")
            fusion.tsdf_fusion(args.data_folder, output_ply, 
                             voxel_size=args.voxel_size,
                             limit=args.limit)
        elif args.fusion_method == 'probabilistic':
            if args.output_ply:
                output_ply = args.output_ply if (os.path.isabs(args.output_ply) or os.path.dirname(args.output_ply)) else os.path.join(args.output_folder, args.output_ply)
            else:
                output_ply = os.path.join(args.output_folder, "fused_pointcloud_probabilistic.ply")
            fusion.probabilistic_fusion(args.data_folder, output_ply, 
                                      voxel_size=args.voxel_size,
                                      limit=args.limit)

        # 融合后可选自动建模
        if args.model and output_ply is not None:
            mesh_out = os.path.join(args.output_folder,
                                    os.path.basename(output_ply).replace('.ply', '_mesh.ply'))
            ok = PointCloudModeler.reconstruct_from_ply(
                input_ply=output_ply,
                output_mesh=mesh_out,
                method=args.model_method,
                downsample_voxel=args.model_downsample_voxel,
                estimate_normals=True,
                poisson_depth=args.model_poisson_depth,
                bpa_radius=args.model_bpa_radius,
                alpha=args.model_alpha,
                simplify_faces=args.model_simplify_faces,
            )
            if not ok:
                print("自动建模失败，已跳过。")
    
    if not args.visualize and not args.fuse:
        print("请指定操作类型：--visualize 或 --fuse 或两者都指定")
        print("示例用法:")
        print("  # 可视化DMB文件")
        print("  python recon-lidar.py --visualize --data_folder data/data --output_folder output")
        print("  # 简单点云融合")
        print("  python recon-lidar.py --fuse --fusion-method simple --data_folder data/data --output_folder output")
        print("  # 几何一致性融合")
        print("  python recon-lidar.py --fuse --fusion-method consistency --data_folder data/data --output_folder output --limit 5")
        print("  # TSDF体素融合")
        print("  python recon-lidar.py --fuse --fusion-method tsdf --data_folder data/data --output_folder output --limit 5")
        print("  # 概率式贝叶斯融合")
        print("  python recon-lidar.py --fuse --fusion-method probabilistic --data_folder data/data --output_folder output --limit 5")


if __name__ == "__main__":
    main()
