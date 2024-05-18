# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifoed the original code of exporter.py to export Gaussian splats from nerfgs

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

import pymeshlab
import torch
from jaxtyping import Float
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor
from pathlib import Path
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.scripts.exporter import Exporter, validate_pipeline

if TYPE_CHECKING:
    # Importing open3d can take ~1 second, so only do it below if we actually
    # need it.
    import open3d as o3d

import cv2

def generate_nerfgs(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = False,
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        reorient_normals: Whether to re-orient the normals based on the view direction.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    spherocal_harmonics = []
    densities = []
    points = []
    rgbs = []
    normals = []
    view_directions = []
    if use_bounding_box and (crop_obb is not None and bounding_box_max is not None):
        CONSOLE.print("Provided aabb and crop_obb at the same time, using only the obb", style="bold yellow")
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            normal = None
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            
            # HSV filtering of sky
            rgb_img = (rgba.cpu().numpy()[:,np.newaxis,:3] * 255).astype(np.uint8)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
           
            lower = np.array([80,0, 216])
            upper = np.array([150, 255, 255])

            # Create HSV Image and threshold into a range.
            mask_hsv = (cv2.inRange(hsv_img, lower, upper) == 0)
            mask_hsv = torch.tensor(mask_hsv, dtype=torch.bool).to(rgba.device)
            
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.75
            
            mask = torch.logical_and(mask, mask_hsv.squeeze())
                  
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if use_bounding_box:
                if crop_obb is None:
                    comp_l = torch.tensor(bounding_box_min, device=point.device)
                    comp_m = torch.tensor(bounding_box_max, device=point.device)
                    assert torch.all(
                        comp_l < comp_m
                    ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                    mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
                else:
                    mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]
            
            points.append(point)
            rgbs.append(rgb)
            view_directions.append(view_direction)
            if normal is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    view_directions = torch.cat(view_directions, dim=0).cpu()

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
    
    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            view_directions = view_directions[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    # re-orient the normals
    if reorient_normals:
        normals = torch.from_numpy(np.array(pcd.normals)).float()
        mask = torch.sum(view_directions * normals, dim=-1) > 0
        normals[mask] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    
    filtered_points = torch.from_numpy(np.asarray(pcd.points)).to(points)
    density, shs = pipeline.model.field.get_spherical_harmonics(filtered_points)
        
    return filtered_points, density, shs

@dataclass
class ExportNerfGS(Exporter):
    """Export NeRFGS."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "open3d"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = True
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
      """
          Find k-nearest neighbors using sklearn's NearestNeighbors.
      x: The data tensor of shape [num_samples, num_features]
      k: The number of neighbors to retrieve
      """
      # Convert tensor to numpy array
      x_np = x

      # Build the nearest neighbors model
      from sklearn.neighbors import NearestNeighbors

      nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

      # Find the k-nearest neighbors
      distances, indices = nn_model.kneighbors(x_np)

      # Exclude the point itself from the result and return
      return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
  
    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        position, density, spherical_harmonics = generate_nerfgs(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {position.shape[0]}")

        nerfgs_filename = self.output_dir / "nerfgs.ply"
        
        nerfgs_data = {}
        with torch.no_grad():
          position = position.cpu().numpy()
          nerfgs_data["positions"] = position
          nerfgs_data["normals"] = np.zeros_like(position, dtype=np.float32)

          spherical_harmonics = spherical_harmonics.cpu().numpy()

          for i in range(spherical_harmonics.shape[1]):
              if i < pipeline.model.config.sh_degree:
                  nerfgs_data[f"f_dc_{i}"] = spherical_harmonics[:,i, None]
              else:
                  reformat_sh = spherical_harmonics[:, pipeline.model.config.sh_degree:]
                  reformat_sh = reformat_sh.reshape(spherical_harmonics.shape[0], 3, -1)
                  reformat_sh = np.swapaxes(reformat_sh, 1, 2)
                  reformat_sh = reformat_sh.reshape(spherical_harmonics.shape[0], -1)
                  nerfgs_data[f"f_rest_{i - pipeline.model.config.sh_degree}"] = spherical_harmonics[:,i, None]

          nerfgs_data["opacity"] = density.cpu().numpy()

          distances, _ = self.k_nearest_sklearn(position, 3)
          distances = torch.from_numpy(distances)
          # find the average of the three nearest neighbors for each point and use that as the scale
          avg_dist = distances.mean(dim=-1, keepdim=True)
          avg_dist = torch.min(avg_dist, torch.quantile(avg_dist, 0.8))
          scales = torch.log(avg_dist.repeat(1, 3)).cpu().numpy()
          for i in range(3):
                nerfgs_data[f"scale_{i}"] = scales[:, i, None]

          N = position.shape[0]
          quats = np.hstack(
              [
                  np.zeros((N, 1)),
                  np.zeros((N, 1)),
                  np.zeros((N, 1)),
                  np.ones((N, 1)),
              ]
          )

          for i in range(4):
                nerfgs_data[f"rot_{i}"] = quats[:, i, None]

        pcd = o3d.t.geometry.PointCloud(nerfgs_data)

        o3d.t.io.write_point_cloud(str(nerfgs_filename), pcd)

        CONSOLE.print("[bold green]:white_check_mark: Saving Spherical Harmonics")

Commands = tyro.conf.FlagConversionOff[
    Union[
          Annotated[ExportNerfGS, tyro.conf.subcommand(name="NerfGS")],
    ]
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
