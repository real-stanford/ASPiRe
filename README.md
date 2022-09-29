# ASPiRe

[Mengda Xu](https://www.cs.columbia.edu/~shurans/)
[Shuran Song](https://www.cs.columbia.edu/~shurans/)
<br>
Columbia University
<br>
Neural Information Processing Systems / NeurIPS 2022

### [Project Page](https://aspire.cs.columbia.edu//)

<!-- | [arXiv](https://arxiv.org/abs/2109.05668) -->

## Overview

This repo contains the PyTorch implementation for paper "ASPiRe: Adaptive Skill Priors for Reinforcement Learning".

<!-- ![teaser](figures/teaser.jpg) -->

## Content

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Installation

```sh
cd path/to/ASPiRe
conda env create -f environment.yml
source activate aspire
pip install -e .
```

## Data Preparation

We provide the data for learning the skill priors for point maze.

- [navigation_dataset](https://drive.google.com/file/d/1VP0nsQqCGE0H-6e5dTVgesL0Ua4ZsTt1/view?usp=sharing): The point mass agent navigates in medium size mazes.
- [avoid_dataset](https://drive.google.com/file/d/1ND2rmKYirsH4XHnsJNW7ks_tkdc6ywxl/view?usp=sharing): The point mass agent avoids the obstacale in front.

## Training

Learning skill priors

<!-- Hyper-parameters mentioned in paper are provided in default arguments. -->

```sh
python script/train_prior.py --d1 NAV_DATA_PATH --d2 AVOID_DATA_PATH --log --kl_analytic --use_batch_norm --name PRIOR_NAME
```

A directory will be created at `skill_prior/maze/PRIOR_NAME`, in which checkpoints will be stored.

Learning downstrem task

<!-- Hyper-parameters mentioned in paper are provided in default arguments. -->

```sh
python script/train_maze.py --prior_name PRIOR_NAME  --prior_checkpoint PRIOR_CHECKPOINT --analytic_kl --raw_kl --use_batch_norm --weight_use_batch_norm --name EXP_NAME
```

A directory will be created at `Experiment/EXP_NAME`, in which checkpoints will be stored.

## BibTeX

```

```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement
