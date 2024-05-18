import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfgs.nerfgs_model import nerfgsModelConfig, nerfgsModel

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.utils import profiler

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
import cv2
import numpy as np
import torch
import os

import json
from pathlib import Path
from time import time

from ipdb import set_trace as st

@dataclass
class nerfgsPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: nerfgsPipeline)
    """target class to instantiate"""
    # datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=nerfgsModelConfig)
    """specifies the model config"""
    export_nerf_gs_data: bool = False

class nerfgsPipeline(VanillaPipeline):
  def __init__(
    self,
    config: nerfgsPipelineConfig,
    device: str,
    test_mode: Literal["test", "val", "inference"] = "val",
    world_size: int = 1,
    local_rank: int = 0,
    grad_scaler: Optional[GradScaler] = None,
  ):
      super(VanillaPipeline, self).__init__()
      self.config = config
      self.test_mode = test_mode
      self.datamanager: DataManager = config.datamanager.setup(
          device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
      )
      self.datamanager.to(device)

      assert self.datamanager.train_dataset is not None, "Missing input dataset"
      self._model = config.model.setup(
          scene_box=self.datamanager.train_dataset.scene_box,
          num_train_data=len(self.datamanager.train_dataset),
          metadata=self.datamanager.train_dataset.metadata,
          device=device,
          grad_scaler=grad_scaler,
      )
      self.model.to(device)

      self.world_size = world_size
      if world_size > 1:
          self._model = typing.cast(
              nerfgsModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
          )
          dist.barrier(device_ids=[local_rank])
  def filter_sky_pixels(self, images_dict: Dict[str, torch.Tensor], output: torch.Tensor) -> None:
    """Filter out the sky pixels from the images_dict.

    Args:
        images_dict: dictionary of images

    Returns:
        images_dict: dictionary of images with sky pixels removed
    """
    rgb = images_dict['image']
    rgb_img = (rgb.cpu().numpy() * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([80,0, 216])
    upper = np.array([150, 255, 255])

    # Create HSV Image and threshold into a range.
    mask_hsv = (cv2.inRange(hsv_img, lower, upper) == 255)
    mask_hsv = torch.tensor(mask_hsv, dtype=torch.bool).to(rgb.device)
    rgb[mask_hsv] = 0.0
    output[mask_hsv] = 0.0
    
    # return self.model.filter_sky_pixels(images_dict)

  @profiler.time_function
  def get_average_eval_image_metrics(
      self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
  ):
      """Iterate over all the images in the eval dataset and get the average.

      Args:
          step: current training step
          output_path: optional path to save rendered images to
          get_std: Set True if you want to return std with the mean metric.

      Returns:
          metrics_dict: dictionary of metrics
      """
      
      self.eval()
      metrics_dict_list = []
      assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
      num_images = len(self.datamanager.fixed_indices_eval_dataloader)
      with Progress(
          TextColumn("[progress.description]{task.description}"),
          BarColumn(),
          TimeElapsedColumn(),
          MofNCompleteColumn(),
          transient=True,
      ) as progress:
          task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
          
          if self.config.export_nerf_gs_data:
              camera_data = self.datamanager.fixed_indices_eval_dataloader[0][0]
              transforms_json = {}
              transforms_json['fl_x'] = camera_data.fx.item()
              transforms_json['fl_y'] = camera_data.fy.item()
              transforms_json['cx'] = camera_data.cx.item()
              transforms_json['cy'] = camera_data.cy.item()
              transforms_json['w'] = camera_data.width.item()
              transforms_json['h'] = camera_data.height.item()
              transforms_json['camera_model'] = "SIMPLE_PINHOLE"
              transforms_json['frames'] = []

              if not os.path.exists(output_path):
                  os.makedirs(output_path)
              
              image_path = os.path.join(output_path, "images")
              
              if not os.path.exists(image_path):
                  os.makedirs(image_path)
          
          for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
              if self.config.export_nerf_gs_data:
                  rgb_img = (batch['image'].cpu().numpy() * 255).astype(np.uint8)
                  bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                  cv2.imwrite(image_path + f"/frame_eval_{batch['image_idx']:05d}.jpg", bgr_img)  
              
                  image_data = {}
                  image_data['file_path'] = f"images/frame_eval_{batch['image_idx']:05d}.jpg"
                  transform_lst = camera.camera_to_worlds[0].cpu().numpy().tolist()
                  transform_lst.append([0, 0, 0, 1])
                  image_data['transform_matrix'] = transform_lst                                     
                  transforms_json['frames'].append(image_data)
              
              # time this the following line
              inner_start = time()
              outputs = self.model.get_outputs_for_camera(camera=camera)
              height, width = camera.height, camera.width
              num_rays = height * width
                              
              self.filter_sky_pixels(batch, outputs['rgb'])
                              
              metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
              
              assert "num_rays_per_sec" not in metrics_dict
              metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
              fps_str = "fps"
              assert fps_str not in metrics_dict
              metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
              metrics_dict_list.append(metrics_dict)
              progress.advance(task)
          
          if self.config.export_nerf_gs_data:
              for camera, batch in self.datamanager.fixed_indices_train_dataloader:
                  # time this the following line
                  outputs = self.model.get_outputs_for_camera(camera=camera)
                  
                  # save images
                  rgb_img = (outputs['rgb'].cpu().numpy() * 255).astype(np.uint8)
                  bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                                  
                  cv2.imwrite(image_path + f"/frame_train_{batch['image_idx']:05d}.jpg", bgr_img) 
                  
                  image_data = {}
                  image_data['file_path'] = f"images/frame_train_{batch['image_idx']:05d}.jpg"
                  transform_lst = camera.camera_to_worlds[0].cpu().numpy().tolist()
                  transform_lst.append([0, 0, 0, 1])
                  image_data['transform_matrix'] = transform_lst
                  transforms_json['frames'].append(image_data)                                               
              
              with open(str(output_path) + '/transforms.json', 'w') as f:
                  json.dump(transforms_json, f, indent=4)    
      # average the metrics list
      metrics_dict = {}
      for key in metrics_dict_list[0].keys():
          if get_std:
              key_std, key_mean = torch.std_mean(
                  torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
              )
              metrics_dict[key] = float(key_mean)
              metrics_dict[f"{key}_std"] = float(key_std)
          else:
              metrics_dict[key] = float(
                  torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
              )
      self.train()
      return metrics_dict
