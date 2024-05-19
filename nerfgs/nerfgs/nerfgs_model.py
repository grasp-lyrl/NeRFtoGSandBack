from nerfstudio.models.splatfacto import *
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter

from dataclasses import dataclass, field
from typing import Type, Dict, List, Optional

import numpy as np
import torch

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from plyfile import PlyData


@dataclass
class nerfgsModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: nerfgsModel)
    load_ply: bool = True
    ply_file_path: str = None


class nerfgsModel(SplatfactoModel):
    config: nerfgsModelConfig

    def populate_modules(self):
        if self.config.load_ply:
            if self.seed_points is not None and not self.config.random_init:
                self.means = torch.nn.Parameter(
                    self.seed_points[0]
                )  # (Location, Color)
            else:
                self.means = torch.nn.Parameter(
                    (torch.rand((self.config.num_random, 3)) - 0.5)
                    * self.config.random_scale
                )

            self.max_2Dsize = None
            self.xys_grad_norm = None
            self.load_state_dict_from_ply(dict)

            self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
                num_cameras=self.num_train_data, device="cpu"
            )

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
        old_vertices = ply_data["vertex"]
        device = self.means.device
        # Remove bounding box
        vertices = old_vertices  # self.remove_bounding_box(old_vertices, device)

        num_splats = len(vertices)

        means = torch.stack(
            (
                torch.tensor(vertices["x"], device=device),
                torch.tensor(vertices["y"], device=device),
                torch.tensor(vertices["z"], device=device),
            ),
            dim=1,
        ).float()
        scales = torch.stack(
            (
                torch.tensor(vertices["scale_0"], device=device),
                torch.tensor(vertices["scale_1"], device=device),
                torch.tensor(vertices["scale_2"], device=device),
            ),
            dim=1,
        ).float()
        quats = torch.stack(
            (
                torch.tensor(vertices["rot_0"], device=device),
                torch.tensor(vertices["rot_1"], device=device),
                torch.tensor(vertices["rot_2"], device=device),
                torch.tensor(vertices["rot_3"], device=device),
            ),
            dim=1,
        ).float()
        opacities = (
            torch.tensor(vertices["opacity"], device=device)
            .reshape(num_splats, 1)
            .float()
        )

        features_dc = torch.stack(
            (
                torch.tensor(vertices["f_dc_0"], device=device),
                torch.tensor(vertices["f_dc_1"], device=device),
                torch.tensor(vertices["f_dc_2"], device=device),
            ),
            dim=1,
        ).float()

        f_rests = torch.zeros(
            num_splats, num_sh_bases(self.config.sh_degree) - 1, 3, device=device
        ).float()

        for j in range(3):
            f_rest_portion = torch.zeros(
                num_splats, num_sh_bases(self.config.sh_degree) - 1, device=device
            )
            for i in range(num_sh_bases(self.config.sh_degree) - 1):
                f_rest_portion[:, i] = torch.tensor(
                    vertices["f_rest_{}".format(i)], device=device
                )
            f_rests[:, :, j] = f_rest_portion
        print("Means shape: ", means.shape)
        print("Scales shape: ", scales.shape)
        print("Opacity shape: ", opacities.shape)
        print("Quats shape: ", quats.shape)
        print("features dc shape: ", features_dc.shape)
        print("f_rests shape: ", f_rests.shape)

        self.means = torch.nn.Parameter(means)
        self.scales = torch.nn.Parameter(scales)
        self.quats = torch.nn.Parameter(quats)
        self.opacities = torch.nn.Parameter(opacities)
        self.features_dc = torch.nn.Parameter(features_dc)
        self.features_rest = torch.nn.Parameter(f_rests)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "xyz": means,
                "scaling": scales,
                "rotation": self.quats,
                "features_dc": features_dc,
                "features_rest": f_rests,
                "opacity": opacities,
            }
        )

        # super().load_state_dict(dict, **kwargs)

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "xyz",
                "scaling",
                "rotation",
                "features_dc",
                "features_rest",
                "opacity",
            ]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps
