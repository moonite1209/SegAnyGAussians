# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import json
import torch
import os

from tqdm import tqdm
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, render_contrastive_feature, render_semantic_feature
from argparse import ArgumentParser
# from gaussian_renderer import GaussianModel
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

# from scene.gaussian_model import GaussianModel
from scene import GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from saga_data.light_camera import LightCamera
from saga_data.specs import CameraParams

from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig, OmegaConf
from saga_config import GuiAppConfig, GuiConfig, ModelConfig, PipeConfig
from enum import Enum, Flag, auto

from utils.visualization_utils import labels_to_color

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(dx)
        rotvec_y = side * np.radians(dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += np.array([-dx, dy, dz])

class RenderMode(Enum):
    rgb = auto()
    feature = auto()
    semantic_feature = auto()
    cluster = auto()
class FilterMode(Flag):
    none = 0
    label = auto()
    scale = auto()
    opacity = auto()
    weight = auto()
class GUI:
    @property
    def image_height(self):
        return int(self.window_height)
    @property
    def image_width(self):
        return int(self.window_width * 0.9)
    
    def __init__(self, gui_cfg: GuiConfig, feature_gaussians, background_color, background_feature, pipe):
        self.feature_gaussians = feature_gaussians
        self.background_color = background_color
        self.background_feature = background_feature
        self.pipe = pipe
        self.window_height = gui_cfg.window_height
        self.window_width = gui_cfg.window_width
        self.orbit_camera = OrbitCamera(self.image_width, self.image_height)
        self.should_update_image = True
        self.render_mode = RenderMode.rgb
        self.filter_mode = FilterMode.none
        self.label = None
        self.point_label, self.cluster_class = self.load_file(gui_cfg.json_path)
        self.pca = self.load_pca()
        self.override_color = labels_to_color(torch.tensor(self.point_label)).to('cuda')

        self.construct_gui()
        dpg.show_viewport()
        # dpg.start_dearpygui()
        while dpg.is_dearpygui_running():
            self.before_render()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def before_render(self):
        if self.should_update_image:
            self.update_image()
            self.should_update_image = False

    def construct_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Viewer', vsync=True, width=self.window_width, height=self.window_height)
        dpg.setup_dearpygui()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.image_width, self.image_height, np.random.randn(self.image_width,self.image_height,3).flatten(), tag="texture", format=dpg.mvFormat_Float_rgb)

        def render_mode_handler(sender, app_data, user_data):
            if app_data == 'rgb':
                self.render_mode = RenderMode.rgb
            elif app_data == 'feature':
                self.render_mode = RenderMode.feature
            elif app_data == 'semantic_feature':
                self.render_mode = RenderMode.semantic_feature
            elif app_data == 'cluster':
                self.render_mode = RenderMode.cluster
            self.should_update_image = True
        def filter_mode_handler(sender, app_data, user_data):
            if sender == 'label_checker':
                if app_data:
                    self.filter_mode |= FilterMode.label
                else:
                    self.filter_mode &= ~FilterMode.label
            if sender == 'scale_checker':
                if app_data:
                    self.filter_mode |= FilterMode.scale
                else:
                    self.filter_mode &= ~FilterMode.scale
            if sender == 'opacity_checker':
                if app_data:
                    self.filter_mode |= FilterMode.opacity
                else:
                    self.filter_mode &= ~FilterMode.opacity
            if sender == 'weight_checker':
                if app_data:
                    self.filter_mode |= FilterMode.weight
                else:
                    self.filter_mode &= ~FilterMode.weight
            self.should_update_image = True
        def cluster_select_handler(sender, app_data, user_data):
            self.label = int(app_data.split(maxsplit=1)[0])
            self.should_update_image = True
        with dpg.window(tag="primary_window", no_scrollbar=True):
            with dpg.group(horizontal=True):
                with dpg.group(tag='group1'):
                    dpg.add_image("texture", tag='image')
                with dpg.group(tag='group2'):
                    dpg.add_listbox(['rgb', 'feature', 'semantic_feature', 'cluster'], label='mode', tag='mode_selector', default_value='rgb', callback=render_mode_handler)
                    dpg.add_checkbox(label="Label", tag='label_checker', callback=filter_mode_handler)
                    dpg.add_checkbox(label="Scale", tag='scale_checker', callback=filter_mode_handler)
                    dpg.add_checkbox(label="Opacity", tag='opacity_checker', callback=filter_mode_handler)
                    dpg.add_checkbox(label="Weight", tag='weight_checker', callback=filter_mode_handler)
                    dpg.add_listbox([f"{k} {v['class']}" for k,v in self.cluster_class.items()], label='cluster', tag='cluster_selector', num_items=10, callback=cluster_select_handler)
        dpg.set_primary_window("primary_window", True)
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("primary_window", theme_no_padding)

        def mouse_wheel_handler(sender, app_data, user_data):
            if not dpg.is_item_hovered('group1'):
                return
            delta = app_data
            if delta == 0:
                return
            self.orbit_camera.scale(0.1*delta)
            self.should_update_image = True
            # self.update_image()
        def mouse_left_click_handler(sender, app_data, user_data):
            if not dpg.is_item_hovered('group1'):
                return
            user_data['is_clicked'] = True
        def mouse_left_move_handler(sender, app_data, user_data):
            if not dpg.is_item_hovered('group1') or not user_data['is_clicked']:
                return
            last_x, last_y = user_data['last_x'], user_data['last_y']
            user_data['last_x'], user_data['last_y'] = app_data
            if last_x is None or last_y is None:
                return
            x, y = app_data
            dx, dy = x - last_x, y - last_y
            if dx == 0 and dy == 0:
                return
            self.orbit_camera.orbit(0.1*dx, -0.1*dy)
            self.should_update_image = True
            # self.update_image()
        def mouse_left_release_handler(sender, app_data, user_data):
            user_data['is_clicked'] = False
            user_data['last_x'] = user_data['last_y'] = None
        def mouse_right_click_handler(sender, app_data, user_data):
            if not dpg.is_item_hovered('group1'):
                return
            user_data['is_clicked'] = True
        def mouse_right_move_handler(sender, app_data, user_data):
            if not dpg.is_item_hovered('group1') or not user_data['is_clicked']:
                return
            last_x, last_y = user_data['last_x'], user_data['last_y']
            user_data['last_x'], user_data['last_y'] = app_data
            if last_x is None or last_y is None:
                return
            x, y = app_data
            dx, dy = x - last_x, y - last_y
            if dx == 0 and dy == 0:
                return
            self.orbit_camera.pan(0.001*dx, -0.001*dy)
            self.should_update_image = True
            # self.update_image()
        def mouse_right_release_handler(sender, app_data, user_data):
            user_data['is_clicked'] = False
            user_data['last_x'] = user_data['last_y'] = None

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=mouse_wheel_handler)
            left_status = {'is_clicked': False, 'last_x': None, 'last_y': None}
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=mouse_left_click_handler, user_data=left_status)
            dpg.add_mouse_move_handler(callback=mouse_left_move_handler, user_data=left_status)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=mouse_left_release_handler, user_data=left_status)
            right_status = {'is_clicked': False, 'last_x': None, 'last_y': None}
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=mouse_right_click_handler, user_data=right_status)
            dpg.add_mouse_move_handler(callback=mouse_right_move_handler, user_data=right_status)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Right, callback=mouse_right_release_handler, user_data=right_status)
            
        def resize_handler(sender, app_data, user_data):
            self.window_width, self.window_height = dpg.get_item_state(app_data)['rect_size']
            self.update_item_size()
        with dpg.item_handler_registry(tag='handlers'):
            dpg.add_item_resize_handler(tag='resize_handler', callback=resize_handler)
        dpg.bind_item_handler_registry("primary_window", "handlers")

    def load_pca(self):
        pca = PCA(n_components=3)
        return pca.fit(self.feature_gaussians.get_instance_features.detach().cpu().numpy())

    def load_file(self, path):
        if not path:
            return None, None
        with open(path, 'r') as f:
            file = json.load(f)
        point_label = file.get('point_labels', None)
        cluster_class = file.get('instances', None)
        if cluster_class:
            cluster_class = {int(k): v for k, v in cluster_class.items()}
        return point_label, cluster_class

    def update_item_size(self):
        dpg.configure_item("image", width=self.image_width, height=self.image_height)
        dpg.configure_item("group2", width=self.window_width-self.image_height)

    def update_image(self):
        image = self.render()
        dpg.set_value("texture", image.flatten())

    def construct_camera(self) -> LightCamera:
        if self.orbit_camera.rot_mode == 1:
            pose = self.orbit_camera.pose_movecenter
        elif self.orbit_camera.rot_mode == 0:
            pose = self.orbit_camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.orbit_camera.fovy * ss

        fy = fov2focal(fovy, self.image_height)
        fx = fy
        params = CameraParams(
            R=R,
            T=t,
            uid=0,
            width=self.image_width,
            height=self.image_height,
            fx=fx,
            fy=fy,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
        )
        return LightCamera.from_params(params)

    def filter_mask(self, gaussians: FeatureGaussianModel):
        mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        if self.filter_mode & FilterMode.label:
            mask &= torch.tensor(self.point_label, dtype=torch.long, device="cuda")==self.label
        if self.filter_mode & FilterMode.scale:
            mask &= gaussians.get_scaling.max(dim=-1).values<gaussians.get_scaling.max(dim=-1).values.median()*0.8
        if self.filter_mode & FilterMode.opacity:
            ...
        if self.filter_mode & FilterMode.weight:
            ...
        return mask

    def render(self):
        camera = self.construct_camera()
        camera.to('cuda')
        filter_mask = self.filter_mask(self.feature_gaussians)
        with torch.inference_mode():
            if self.render_mode == RenderMode.rgb:
                rendered_image = render(camera, self.feature_gaussians, self.pipe, self.background_color, filtered_mask=~filter_mask)['render'].detach().permute(1,2,0).cpu().numpy()
            elif self.render_mode == RenderMode.feature:
                rendered_feature = render_contrastive_feature(camera, self.feature_gaussians, self.pipe, self.background_feature, filtered_mask=~filter_mask)['render'].detach().permute(1,2,0).cpu().numpy()
                H,W,C = rendered_feature.shape
                rendered_image = self.pca.transform(rendered_feature.reshape(-1,C))
                rendered_image = minmax_scale(rendered_image, (0,1)).reshape(H,W,3)
            elif self.render_mode == RenderMode.semantic_feature:
                rendered_feature = render_semantic_feature(camera, self.feature_gaussians, self.pipe, self.background_feature, filtered_mask=~filter_mask)['render'].detach().permute(1,2,0).cpu().numpy()
                H,W,C = rendered_feature.shape
                rendered_image = self.pca.transform(rendered_feature.reshape(-1,C))
                rendered_image = minmax_scale(rendered_image, (0,1)).reshape(H,W,3)
            elif self.render_mode == RenderMode.cluster:
                rendered_image = render(camera, self.feature_gaussians, self.pipe, self.background_color, override_color=self.override_color, filtered_mask=~filter_mask)['render'].detach().permute(1,2,0).cpu().numpy()
        torch.cuda.empty_cache()
        return rendered_image
def load_model(gui_cfg: GuiConfig, model: ModelConfig):
    feature_gaussians = FeatureGaussianModel(model.sh_degree, model.instance_feature_dim, model.semantic_feature_dim)
    feature_gaussians.load_ply(gui_cfg.feature_point_cloud_path)
    feature_gaussians.eval()
    background_color = torch.tensor([1.]*3 if model.white_background else [0.]*3, dtype=torch.float32, device="cuda")
    background_feature = torch.tensor([0.]*model.instance_feature_dim, dtype=torch.float32, device="cuda")
    return feature_gaussians, background_color, background_feature

@hydra.main(config_path="configs", config_name="gui", version_base=None)
def main(cfg: DictConfig):
    app_cfg = GuiAppConfig(**OmegaConf.to_container(cfg, resolve=True))
    model: ModelConfig = app_cfg.model
    pipe: PipeConfig = app_cfg.pipe
    gui_cfg: GuiConfig = app_cfg.gui

    safe_state(gui_cfg.quiet)
    feature_gaussians, background_color, background_feature = load_model(gui_cfg, model)
    GUI(gui_cfg, feature_gaussians, background_color, background_feature, pipe)

if __name__ == "__main__":
    main()
