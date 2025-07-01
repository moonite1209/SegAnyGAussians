import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from arguments import ModelParams
from utils.camera_utils import loadCam

class CameraDataset(Dataset):
    def __init__(self, args: ModelParams, camera_infos, resolution_scale = 1.0):
        self.camera_infos = camera_infos
        self.args = args
        self.resolution_scale = resolution_scale

    def __len__(self):
        return len(self.camera_infos)
    
    def __getitem__(self, idx):
        camera_info = self.camera_infos[idx]
        return loadCam(self.args, idx, camera_info, self.resolution_scale)