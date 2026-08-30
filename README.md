# TSDGFlow

TSDGFlow is an unsupervised method for multivariate time-series anomaly
detection.

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
  --freq_patch_size 16 --freq_patch_stride 8 \
  --spectral_gating_mode mlp \
  --graph_fusion_mode fixed_alpha --graph_alpha 0.7 \
  --x3_align_lambda 0.01 --x3_align_direction t2f
```

## Acknowledgements

This implementation is derived from the public USD codebase, which is itself
based on MTGFlow. Please cite those projects together with TSDGFlow when using
this code.
