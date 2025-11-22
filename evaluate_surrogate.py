"""
Benchmark and sanity-check for the transformer halo surrogate.

Usage example:
    python evaluate_surrogate.py \
        --model products/halo_transformer.pt \
        --n-samples 200 \
        --outdir products
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

SIDM_PATH = "/Users/chengzhangjiang/czjiang/personal/python/SIDM"
if SIDM_PATH not in sys.path:
    sys.path.insert(0, SIDM_PATH)

import profiles as pr  # noqa: E402
import galhalo as gh  # noqa: E402

from halo_surrogate_torch import (
    ParameterSampler,
    HaloSurrogateTorch,
    calculate_log_c200_mean,
)

H0 = 67.66
SIGMAMX = 1
TAGE = 10
R_FULLRANGE = np.linspace(0.01, 20, 512)


def calc_contracted_params(log_r0, log_mbar, log_m200, log_c200) -> Tuple[float, float, float]:
    r0 = 10.0**log_r0
    mbar = 10.0**log_mbar
    m200 = 10.0**log_m200
    c200 = 10.0**log_c200
    halo_init = pr.NFW(m200, c200, Delta=200.0, z=0.0)
    disk = pr.Hernquist(mbar, r0)
    MvD, cD, aD = gh._contra(R_FULLRANGE, halo_init, disk)
    return np.log10(MvD), cD, aD


def calc_core_params(log_r0, log_mbar, log_m200, log_c200) -> Tuple[float, float]:
    r0 = 10.0**log_r0
    mbar = 10.0**log_mbar
    m200 = 10.0**log_m200
    c200 = 10.0**log_c200
    halo_init = pr.NFW(m200, c200, Delta=200.0, z=0.0)
    disk = pr.Hernquist(mbar, r0)
    halo_contra = gh.contra(R_FULLRANGE, halo_init, disk)[0]
    r1 = pr.r1(halo_contra, sigmamx=SIGMAMX, tage=TAGE)
    rhodm0, sigma0 = pr._stitchSIDMcore(r1, halo_contra, disk)
    return rhodm0, sigma0


def vc_from_dekel(log_MvD: float, cD: float, aD: float, r: np.ndarray) -> np.ndarray:
    halo = pr.Dekel(10**log_MvD, cD, aD)
    return halo.Vcirc(r)


def benchmark(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = HaloSurrogateTorch.from_file(args.model, device=device)
    sampler = ParameterSampler(
        log_m200_range=tuple(args.log_m200_range),
        log_mbar_min=args.log_mbar_min,
        min_delta_m=args.min_delta_m,
        r0_range_kpc=tuple(args.r0_range),
        c_sigma=args.c_sigma,
        seed=args.seed,
    )
    params = sampler.sample(args.n_samples)

    # Direct evaluation
    t0 = time.time()
    direct_contra = []
    direct_core = []
    params_success = []
    for row in params:
        try:
            direct_contra.append(calc_contracted_params(*row))
            direct_core.append(calc_core_params(*row))
            params_success.append(row)
        except Exception:
            # Skip failed physical evaluations
            continue
    direct_time = time.time() - t0
    if not direct_contra:
        raise RuntimeError("No successful direct evaluations; relax priors or reduce samples.")
    direct_contra = np.asarray(direct_contra)
    direct_core = np.asarray(direct_core)
    params_success = np.asarray(params_success)

    # Surrogate
    t1 = time.time()
    pred_contra, pred_core = surrogate.predict(params_success)
    surrogate_time = time.time() - t1
    # Ensure equal lengths in case any unexpected mismatch arises
    n_compare = min(len(direct_contra), len(pred_contra), len(direct_core), len(pred_core))
    direct_contra = direct_contra[:n_compare]
    direct_core = direct_core[:n_compare]
    pred_contra = pred_contra[:n_compare]
    pred_core = pred_core[:n_compare]

    # Errors
    contra_err = pred_contra - direct_contra
    core_err = pred_core - direct_core

    print(f"Direct eval time for {len(direct_contra)} samples: {direct_time:.2f} s")
    print(f"Surrogate eval time for {len(pred_contra)} samples: {surrogate_time:.4f} s")
    print(f"Speedup factor: {direct_time / max(surrogate_time, 1e-6):.1f}x")
    print(
        "Median abs errors (log_MvD, cD, aD):",
        np.median(np.abs(contra_err), axis=0),
    )
    print(
        "Median abs errors (rhodm0, sigma0):",
        np.median(np.abs(core_err), axis=0),
    )

    os.makedirs(args.outdir, exist_ok=True)

    # Scatter residuals (pred vs true), one panel per parameter
    labels_contra = ["log_MvD", "cD", "aD"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, label in enumerate(labels_contra):
        ax = axes[i]
        ax.scatter(direct_contra[:, i], pred_contra[:, i], s=8, alpha=0.5)
        lo = min(direct_contra[:, i].min(), pred_contra[:, i].min())
        hi = max(direct_contra[:, i].max(), pred_contra[:, i].max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Pred {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "surrogate_pred_vs_true_contracted.png"), dpi=200)

    labels_core = ["rhodm0", "sigma0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, label in enumerate(labels_core):
        ax = axes[i]
        ax.scatter(direct_core[:, i], pred_core[:, i], s=8, alpha=0.5)
        lo = min(direct_core[:, i].min(), pred_core[:, i].min())
        hi = max(direct_core[:, i].max(), pred_core[:, i].max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Pred {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "surrogate_pred_vs_true_core.png"), dpi=200)

    # Residual plots (surrogate - direct)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, label in enumerate(labels_contra):
        ax = axes[i]
        ax.scatter(direct_contra[:, i], contra_err[:, i], s=8, alpha=0.5)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel("Surrogate - Direct")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "surrogate_residuals_contracted.png"), dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, label in enumerate(labels_core):
        ax = axes[i]
        ax.scatter(direct_core[:, i], core_err[:, i], s=8, alpha=0.5)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel("Surrogate - Direct")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "surrogate_residuals_core.png"), dpi=200)

    # One illustrative Vcirc comparison (use first successful sample)
    if n_compare == 0:
        return
    log_MvD_true, cD_true, aD_true = direct_contra[0]
    log_MvD_pred, cD_pred, aD_pred = pred_contra[0]
    r = np.linspace(0.01, 20, 200)
    vc_true = vc_from_dekel(log_MvD_true, cD_true, aD_true, r)
    vc_pred = vc_from_dekel(log_MvD_pred, cD_pred, aD_pred, r)

    plt.figure(figsize=(6, 4))
    plt.plot(r, vc_true, label="Direct")
    plt.plot(r, vc_pred, label="Surrogate", ls="--")
    plt.xlabel("r [kpc]")
    plt.ylabel("Vcirc [km/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "surrogate_vcirc_example.png"), dpi=200)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate halo transformer surrogate.")
    parser.add_argument("--model", required=True, help="Path to trained surrogate .pt file.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-m200-range", nargs=2, type=float, default=(8.0, 13.0))
    parser.add_argument("--log-mbar-min", type=float, default=6.5)
    parser.add_argument("--r0-range", nargs=2, type=float, default=(0.05, 50.0))
    parser.add_argument("--c-sigma", type=float, default=0.11)
    parser.add_argument("--min-delta-m", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark(args)


if __name__ == "__main__":
    main()
