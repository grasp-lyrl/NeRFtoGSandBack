# From NeRFs to Gaussian Splats, and Back
This is the implementation of [From NeRFs to Gaussian Splats, and Back](https://arxiv.org/abs/2405.09717); An efficient procedure to convert back and forth between NeRF and GS, and thereby get the best of both approaches.

## Installation
This repository follows the nerfstudio method [template](https://github.com/nerfstudio-project/nerfstudio-method-template/tree/main)

### 0. Install Nerfstudio dependencies
Please follow the Nerfstudio [installation guide](https://docs.nerf.studio/quickstart/installation.html)  to create an environment and install dependencies.

### 1. Install the repository
Clone and navigate into this repository. Run the following commands:

`pip install -e nerfgs`

and

`pip install -e splatting`.

Finally, run `ns-install-cli`.

### 2. Check installation
Run `ns-train --help`. You should be able to find two methods, `nerfgs` and `splatting`, in the list of methods.

## Downloading data
You could download the Giannini-Hall and aspen datasets from [this google drive link](https://drive.google.com/drive/folders/19TV6kdVGcmg3cGZ1bNIUnBBMD-iQjRbG). For our new dataset, Wissahickon and Locust-Walk, coming soon.

## NeRFs to Gaussian Splats
### Training nerfgs
Run the following command for training. Replace `DATA_PATH` with the data directory location.

`ns-train nerfgs --data DATA_PATH --pipeline.model.camera-optimizer.mode off `

To train on Wissahickon or Locust-Walk dataset, you need to add `nerfstudio-data --eval-mode filename` to properly split training and validation data, i.e.,

`ns-train nerfgs --data DATA_PATH --pipeline.model.camera-optimizer.mode off nerfstudio-data --eval-mode filename`


### Export splats from nerfgs
Replace `CONFIG_LOCATION` with the location of config file saved after training.

`ns-export-nerfgs --load-config CONFIG_LOCATION --output-dir exports/nerfgs/ --num-points 2000000 --remove-outliers True --normal-method open3d --use_bounding_box False`

### Show exported splats
Replace `DATA_PATH` with the data directory location. You also need to add `nerfstudio-data --eval-mode filename` if train on Wissahickon or Locust-Walk.

`ns-train splatting --data DATA_PATH --max-num-iterations 1 --pipeline.model.ply-file-path exports/nerfgs/nerfgs.ply`

### Finetuning the splats
We reduces the learning rate for finetuning. You also need to add `nerfstudio-data --eval-mode filename` if train on Wissahickon or Locust-Walk.

`ns-train splatting --dataCONFIG_LOCATION --max-num-iterations 5001 --pipeline.model.ply-file-path exports/nerfgs/nerfgs.ply --pipeline.model.sh-degree-interval 0 --pipeline.model.warmup-length 100 --optimizers.xyz.optimizer.lr 0.00001 --optimizers.xyz.scheduler.lr-pre-warmup 0.0000001 --optimizers.xyz.scheduler.lr-final 0.0000001 --optimizers.features-dc.optimizer.lr 0.01 --optimizers.features-rest.optimizer.lr 0.001 --optimizers.opacity.optimizer.lr 0.05 --optimizers.scaling.optimizer.lr 0.01 --optimizers.rotation.optimizer.lr 0.0000000001 --optimizers.camera-opt.optimizer.lr 0.0000000001 --optimizers.camera-opt.scheduler.lr-pre-warmup 0.0000000001 --optimizers.camera-opt.scheduler.lr-final 0.0000000001`

## Gaussian Splats to NeRFs

### Scene modification
Coming soon

### Creating new dataset
In the new dataset, training images are rendered from splats. Replace `CONFIG_LOCATION` with the location of config file saved after training.

`ns-splatting-render --load-config CONFIG_LOCATION --render-output-path exports/splatting_data --export-nerf-gs-data`

### Training on new dataset
`ns-train nerfgs --data exports/splatting_data --pipeline.model.camera-optimizer.mode off nerfstudio-data --eval-mode filename`

## Extending Nerfgs
The conversion from NeRF to GS has inefficiency as mentioned at the discussion section of the paper. We welcome your efforts to reduce the inefficiency! The code for conversion is mainly in `nerfgs/nerfgs/nerfgs_exporter.py`.

## Bibtex
```
@misc{he2024nerfs,
      title={From NeRFs to Gaussian Splats, and Back}, 
      author={Siming He and Zach Osman and Pratik Chaudhari},
      year={2024},
      eprint={2405.09717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
