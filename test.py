"""Evaluate a released TSDGFlow checkpoint on its corresponding dataset."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from Dataset import load_smd_smap_msl, loader_PSM, loader_SWat, loader_WADI
from models.MTGFLOW import MTGFLOWZL


MODEL_CONFIGS = {
    "PSM": {
        "checkpoint": "PSM.pth",
        "seed": 18,
        "batch_size": 128,
        "window_size": 60,
        "stride_size": 10,
        "n_blocks": 1,
        "n_sensor": 25,
        "k": 20,
    },
    "SWaT": {
        "checkpoint": "SWaT.pth",
        "seed": 19,
        "batch_size": 128,
        "window_size": 60,
        "stride_size": 10,
        "n_blocks": 1,
        "n_sensor": 51,
        "k": 10,
    },
    "MSL": {
        "checkpoint": "MSL.pth",
        "seed": 17,
        "batch_size": 128,
        "window_size": 60,
        "stride_size": 10,
        "n_blocks": 2,
        "n_sensor": 55,
        "k": 20,
    },
    "WADI": {
        "checkpoint": "WADI.pth",
        "seed": 19,
        "batch_size": 64,
        "window_size": 60,
        "stride_size": 10,
        "n_blocks": 5,
        "n_sensor": 123,
        "k": 20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=MODEL_CONFIGS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the default checkpoints/<dataset>.pth file.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_loader(dataset: str, config: dict[str, int]):
    common = (
        config["batch_size"],
        config["window_size"],
        config["stride_size"],
        0.6,
    )
    kwargs = {"k": config["k"], "alpha": 0.1, "seed": config["seed"]}
    if dataset == "PSM":
        _, test_loader, n_sensor = loader_PSM("PSM", *common, **kwargs)
    elif dataset == "SWaT":
        _, test_loader, n_sensor = loader_SWat(*common, **kwargs)
    elif dataset == "WADI":
        _, test_loader, n_sensor = loader_WADI(*common, **kwargs)
    else:
        _, test_loader, n_sensor = load_smd_smap_msl("MSL", *common, **kwargs)
    return test_loader, n_sensor


def build_model(config: dict[str, int], n_sensor: int) -> MTGFLOWZL:
    if n_sensor != config["n_sensor"]:
        raise ValueError(f"Expected {config['n_sensor']} sensors, found {n_sensor}")
    return MTGFLOWZL(
        config["n_blocks"],
        1,
        32,
        1,
        config["window_size"],
        n_sensor,
        dropout=0.0,
        model="MAF",
        batch_norm=False,
        use_spectral_graph=True,
        spectral_representation="real_imag",
        drop_dc=False,
        freq_patch_size=16,
        freq_patch_stride=8,
        freq_embed_dim=0,
        freq_dropout=0.1,
        spectral_graph_topk=0,
        spectral_gating_mode="mlp",
        graph_fusion_mode="fixed_alpha",
        graph_alpha=0.7,
    )


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.dataset]
    seed_everything(config["seed"])
    device = torch.device(args.device)

    test_loader, n_sensor = load_test_loader(args.dataset, config)
    model = build_model(config, n_sensor).to(device)
    checkpoint = args.checkpoint or ROOT / "checkpoints" / config["checkpoint"]
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    scores = []
    with torch.no_grad():
        for x, _, _, _ in test_loader:
            _, log_prob, _ = model.test(x.to(device))
            scores.append((-log_prob).cpu().numpy())

    anomaly_scores = np.concatenate(scores)
    labels = np.asarray(test_loader.dataset.label, dtype=int)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {checkpoint}")
    print(f"AUROC: {roc_auc_score(labels, anomaly_scores):.6f}")
    print(f"AUPRC: {average_precision_score(labels, anomaly_scores):.6f}")


if __name__ == "__main__":
    main()
