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

# nerfsh field based on modification of the nerfacto field.

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from gsplat.sh import num_sh_bases, spherical_harmonics


class nerfshField(Field):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        sh_degree: int = 3,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(
            self.num_images, self.appearance_embedding_dim
        )
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res
        self.step = 0

        self.sh_degree = sh_degree
        self.dim_sh_ceof = num_sh_bases(self.sh_degree)

        # self.direction_encoding = SHEncoding(
        #     levels=4,
        #     implementation=implementation,
        # )

        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=2,
            min_freq_exp=0,
            max_freq_exp=2 - 1,
            implementation=implementation,
        )

        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(
                self.num_images, self.transient_embedding_dim
            )
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )
            self.field_head_transient_rgb = TransientRGBFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )
            self.field_head_transient_density = TransientDensityFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(),
                num_classes=num_semantic_classes,
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(
                in_dim=self.mlp_pred_normals.get_out_dim()
            )

        self.mlp_head = MLP(
            in_dim=self.position_encoding.get_out_dim()
            + self.geo_feat_dim
            + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=self.dim_sh_ceof * 3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = (
            ray_samples.frustums.directions
        ) 
        directions_flat = directions.reshape(-1, 3)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = (
                self.mlp_transient(transient_input)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(
                x
            )
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = (
                self.field_head_transient_density(x)
            )

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = (
                self.mlp_semantics(semantics_input)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat(
                [positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1
            )

            x = (
                self.mlp_pred_normals(pred_normals_inp)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        positions = ray_samples.frustums.get_positions()
        positions_flat = self.position_encoding(positions.view(-1, 3))
        h = torch.cat(
            [
                positions_flat,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )

        sh_feature = self.mlp_head(h).to(directions)
        n = self.sh_degree  # min(self.step // 1000, self.sh_degree)
        rgb = spherical_harmonics(
            n, directions_flat, sh_feature.view(-1, self.dim_sh_ceof, 3)
        )
        rgb = torch.clamp(rgb + 0.5, min=0.0).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def get_density_from_position(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(-1, 1 + self.geo_feat_dim)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_spherical_harmonics(self, positions: Tensor) -> Tuple[Tensor, Tensor]:

        with torch.no_grad():
            density, density_embedding = self.get_density_from_position(positions)

        if self.use_average_appearance_embedding:
            embedded_appearance = torch.ones(
                (*positions.shape[:-1], self.appearance_embedding_dim),
                device=positions.device,
            ) * self.embedding_appearance.mean(dim=0)
        else:
            embedded_appearance = torch.zeros(
                (*positions.shape[:-1], self.appearance_embedding_dim),
                device=positions.device,
            )

        positions_feat = self.position_encoding(positions)

        hiddle_feat = torch.cat(
            [
                positions_feat,
                density_embedding.to(positions),
                embedded_appearance.to(positions),
            ],
            dim=-1,
        )

        with torch.no_grad():
            sh_feature = self.mlp_head(hiddle_feat).to(positions)

        # from location gives the sh coefs
        return density, sh_feature
