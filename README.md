# TSDGFlow

TSDGFlow is an unsupervised multivariate time-series anomaly detector built on
USD and MTGFlow. It augments the temporal relation graph with a patch-wise
frequency relation graph, fuses the two views, and uses one-way structural
anchoring during training.

This repository intentionally contains only the core implementation, dataset
loaders, compact training/evaluation entry points, and selected pretrained
models. Experiment sweeps, ablation runners, paper drafts, and raw logs are not
included.

## Installation

Python 3.9 is recommended.

```bash
pip install -r requirements.txt
```

## Data

Place the datasets under `Data/input` using the layout expected by the loaders:

```text
Data/input/
├── PSM/
│   ├── test.csv
│   └── test_label.csv
├── SWaT_Dataset_Attack_v0.csv
├── WADI_attackdata.csv
└── processed/
    ├── MSL_test.pkl
    ├── MSL_test_label.pkl
    └── machine-*_test*.pkl
```

The datasets are not redistributed. Please obtain SWaT/WADI from iTrust,
PSM from the MST-VAE release, MSL from Telemanom, and SMD from OmniAnomaly.

## Evaluate a pretrained model

Four selected models are provided as standalone files in `checkpoints/`:
PSM, SWaT, MSL, and WADI.

```bash
python test.py --dataset PSM
python test.py --dataset SWaT
python test.py --dataset MSL
python test.py --dataset WADI
```

On the first run, the dataset loader creates nearest-neighbor index caches in
`save_near_index/`.

## Train

For example, the PSM configuration can be trained with:

```bash
python train.py \
  --name PSM --seed 18 --epoch 400 --batch_size 128 \
  --window_size 60 --stride_size 10 --n_blocks 1 --k 20 \
  --lr 0.002 --alpha 0.1 \
  --loss_weight_manifold_ne 5 --loss_weight_manifold_po 1 \
  --freq_patch_size 16 --freq_patch_stride 8 \
  --spectral_gating_mode mlp \
  --graph_fusion_mode fixed_alpha --graph_alpha 0.7 \
  --x3_align_lambda 0.01 --x3_align_direction t2f
```

## Acknowledgements

This implementation is derived from the public USD codebase, which is itself
based on MTGFlow. Please cite those projects together with TSDGFlow when using
this code.

The upstream USD repository does not currently publish an explicit software
license. Licensing permission must therefore be confirmed before this staging
copy is made public.
