from nerfstudio.models.splatfacto import *
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter

from dataclasses import dataclass, field
from typing import Type

import torch

from plyfile import PlyData
import ast

@dataclass
class nerfgsModelConfig(SplatfactoModelConfig):
  _target: Type = field(default_factory=lambda: nerfgsModel)
  load_ply: bool = True
  ply_file_path: str = ""
  bottom_remove_box_corner: str = ""
  top_remove_box_corner: str = ""

class nerfgsModel(SplatfactoModel):
  config: nerfgsModelConfig

  def populate_modules(self):
    if self.config.load_ply:
      if self.seed_points is not None and not self.config.random_init:
              self.means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
      else:
          self.means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)

      self.max_2Dsize = None
      self.xys_grad_norm = None
      self.load_state_dict_from_ply(dict)

      # metrics
      from torchmetrics.image import PeakSignalNoiseRatio
      from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

      self.psnr = PeakSignalNoiseRatio(data_range=1.0)
      self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
      self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
      self.step = 0

      self.crop_box: Optional[OrientedBox] = None
      if self.config.background_color == "random":
          self.background_color = torch.rand(3)
      else:
          self.background_color = get_color(self.config.background_color)
    else:
       super().populate_modules()


  def load_state_dict_from_ply(self, dict, **kwargs):  # type: ignore

    # Read from ply file
    ply_data = PlyData.read(self.config.ply_file_path)
    vertices = ply_data['vertex']
    device = self.means.device

    # Remove bounding box if provided 
    if len(self.config.bottom_remove_box_corner) > 0 and len(self.config.top_remove_box_corner) > 0:
      self.config.bottom_remove_box_corner = ast.literal_eval(self.config.bottom_remove_box_corner)
      self.config.top_remove_box_corner = ast.literal_eval(self.config.top_remove_box_corner)
      vertices = self.remove_bounding_box(vertices, device)

    num_splats = len(vertices)

    means = torch.stack((torch.tensor(vertices['x'], device = device),
                          torch.tensor(vertices['y'], device = device),
                          torch.tensor(vertices['z'], device =device)), dim=1).float()
    scales = torch.stack((torch.tensor(vertices['scale_0'], device = device),
                          torch.tensor(vertices['scale_1'], device = device),
                          torch.tensor(vertices['scale_2'], device =device)), dim=1).float()
    quats = torch.stack((torch.tensor(vertices['rot_0'], device = device),
                          torch.tensor(vertices['rot_1'], device = device),
                          torch.tensor(vertices['rot_2'], device =device),
                          torch.tensor(vertices['rot_3'], device = device)), dim=1).float()
    opacities = torch.tensor(vertices['opacity'], device = device).reshape(num_splats, 1).float()
    
    features_dc = torch.stack((torch.tensor(vertices['f_dc_0'], device = device),
                          torch.tensor(vertices['f_dc_1'], device = device),
                          torch.tensor(vertices['f_dc_2'], device =device)), dim=1).float()
    
    f_rests = torch.zeros(num_splats, num_sh_bases(self.config.sh_degree) - 1, 3, device=device).float()
    
    for j in range(3):
        f_rest_portion = torch.zeros(num_splats, num_sh_bases(self.config.sh_degree) - 1, device = device)    
        for i in range(num_sh_bases(self.config.sh_degree) - 1):
            f_rest_portion[:, i] = torch.tensor(vertices['f_rest_{}'.format(i)], device = device)
        f_rests[:, :, j] = f_rest_portion

    self.means = torch.nn.Parameter(means)
    self.scales = torch.nn.Parameter(scales)
    self.quats = torch.nn.Parameter(quats)
    self.opacities = torch.nn.Parameter(opacities)
    self.features_dc = torch.nn.Parameter(features_dc)
    self.features_rest = torch.nn.Parameter(f_rests)

  def remove_bounding_box(self, vertices, device):
        min_x = self.config.bottom_remove_box_corner[0] # 0.6220401063486586
        max_x = self.config.top_remove_box_corner[0] # 0.7601881605528515
        min_y = self.config.bottom_remove_box_corner[1] # -0.5871678744127707
        max_y = self.config.top_remove_box_corner[1]  #-0.4676805579564234
        min_z =  self.config.bottom_remove_box_corner[2] #-0.0712212514966822
        max_z =  self.config.top_remove_box_corner[2] #0.26269306846814755

        means = torch.stack((torch.tensor(vertices['x'], device = device),
                             torch.tensor(vertices['y'], device = device),
                             torch.tensor(vertices['z'], device =device)), dim=1)
        valid_vertices = []
        for mean in means:
            # If inside box then we want to remove it
            if mean[0] <  max_x and mean[0] > min_x and mean[1] < max_y and mean[1] > min_y and mean[2] < max_z and mean[2] > min_z:
                valid_vertices.append(False)
            else:
                valid_vertices.append(True)

        vertices = vertices[valid_vertices]

        return vertices


