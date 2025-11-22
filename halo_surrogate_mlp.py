"""
PyTorch residual-MLP surrogate for contracted and cored halos.

This complements the transformer surrogate with a simpler, strong baseline
tailored for tabular data: a residual multi-layer perceptron with SiLU
activations, layer normalization, dropout, and optional smoothness
regularization. It samples the parameter space via the same Sobol-based
ParameterSampler used elsewhere and learns to predict both contracted
Dekel parameters and SIDM core quantities in one model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Ensure SIDM package is importable.
SIDM_PATH = "/Users/chengzhangjiang/czjiang/personal/python/SIDM"
if SIDM_PATH not in sys.path:
    sys.path.insert(0, SIDM_PATH)

import profiles as pr  # noqa: E402
import galhalo as gh  # noqa: E402

H0 = 67.66
SIGMAMX = 1
TAGE = 10
R_FULLRANGE = np.linspace(0.01, 20, 512)


def calculate_log_c200_mean(log_m200: np.ndarray) -> np.ndarray:
    a, b = 0.905, -0.101
    return a + b * (log_m200 - 12 + np.log10(H0 / 100))


@dataclass
class ParameterSampler:
    """Sobol sampler obeying astrophysical priors (mirrors transformer sampler)."""

    log_m200_range: Tuple[float, float] = (8.0, 13.0)
    log_mbar_min: float = 6.5
    min_delta_m: float = 0.15
    r0_range_kpc: Tuple[float, float] = (0.05, 50.0)
    c_sigma: float = 0.11
    seed: Optional[int] = 42

    def sample(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        engine = SobolEngine(dimension=4, scramble=True, seed=self.seed)
        base = engine.draw(n_samples).numpy()

        log_m200 = (
            self.log_m200_range[0]
            + base[:, 0] * (self.log_m200_range[1] - self.log_m200_range[0])
        )
        log_mbar_max = log_m200 - self.min_delta_m
        log_mbar_lower = np.minimum(self.log_mbar_min, log_mbar_max - 1e-3)
        span = np.maximum(log_mbar_max - log_mbar_lower, 1e-3)
        log_mbar = log_mbar_lower + base[:, 1] * span

        r0 = self.r0_range_kpc[0] + base[:, 2] * (
            self.r0_range_kpc[1] - self.r0_range_kpc[0]
        )
        log_r0 = np.log10(np.clip(r0, 1e-3, None))

        log_c_mean = calculate_log_c200_mean(log_m200)
        delta = (base[:, 3] - 0.5) * 10 * self.c_sigma
        log_c = log_c_mean + delta
        log_c = np.clip(log_c, log_c_mean - 5 * self.c_sigma, log_c_mean + 5 * self.c_sigma)

        return np.column_stack([log_r0, log_mbar, log_m200, log_c])


def _compute_contracted_and_cored(
    params: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    contra_targets = []
    core_targets = []

    for row in tqdm(params, desc="Evaluating halos", leave=False):
        log_r0, log_mbar, log_m200, log_c200 = row
        r0 = 10.0 ** log_r0
        mbar = 10.0 ** log_mbar
        m200 = 10.0 ** log_m200
        c200 = 10.0 ** log_c200

        halo_init = pr.NFW(m200, c200, Delta=200.0, z=0.0)
        disk = pr.Hernquist(mbar, r0)

        try:
            MvD, cD, aD = gh._contra(R_FULLRANGE, halo_init, disk)
            halo_contra = gh.contra(R_FULLRANGE, halo_init, disk)[0]
            r1 = pr.r1(halo_contra, sigmamx=SIGMAMX, tage=TAGE)
            rhodm0, sigma0 = pr._stitchSIDMcore(r1, halo_contra, disk)
        except Exception:
            continue

        features.append([log_r0, log_mbar, log_m200, log_c200])
        contra_targets.append([np.log10(MvD), cD, aD])
        core_targets.append([rhodm0, sigma0])

    if not features:
        raise RuntimeError("No successful halo evaluations. Adjust sampling ranges.")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(contra_targets, dtype=np.float32),
        np.asarray(core_targets, dtype=np.float32),
    )


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + eps
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Iterable[float]]) -> "Standardizer":
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        return cls(mean, std)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HaloMLP(nn.Module):
    """Residual MLP with twin heads for contracted and cored outputs."""

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(depth)],
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.contra_head = nn.Linear(hidden_dim, 3)
        self.core_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.contra_head(h), self.core_head(h)


def _build_dataloaders(
    features: np.ndarray,
    contra: np.ndarray,
    core: np.ndarray,
    batch_size: int,
    val_fraction: float,
) -> Tuple[DataLoader, DataLoader, Dict[str, Standardizer]]:
    x_scaler = Standardizer.fit(features)
    y_contra_scaler = Standardizer.fit(contra)
    y_core_scaler = Standardizer.fit(core)

    x = torch.from_numpy(x_scaler.transform(features))
    y_contra = torch.from_numpy(y_contra_scaler.transform(contra))
    y_core = torch.from_numpy(y_core_scaler.transform(core))

    dataset = TensorDataset(x, y_contra, y_core)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        {"x": x_scaler, "contra": y_contra_scaler, "core": y_core_scaler},
    )


def train_mlp_surrogate(
    *,
    n_samples: int,
    sampler: ParameterSampler,
    batch_size: int = 512,
    epochs: int = 200,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    weight_core: float = 1.0,
    smoothness_sigma: float = 0.05,
    smoothness_weight: float = 0.1,
    physics_weight: float = 0.0,
    physics_max_batch: int = 32,
    hidden_dim: int = 256,
    depth: int = 4,
    dropout: float = 0.1,
    device: Optional[str] = None,
    output_path: str = "halo_mlp.pt",
    quiet: bool = False,
) -> Dict[str, float]:
    params = sampler.sample(n_samples)
    features, contra_targets, core_targets = _compute_contracted_and_cored(params)

    train_loader, val_loader, scalers = _build_dataloaders(
        features, contra_targets, core_targets, batch_size, val_fraction
    )

    model = HaloMLP(
        n_features=features.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 5)
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, y_contra, y_core in train_loader:
            xb = xb.to(device)
            y_contra = y_contra.to(device)
            y_core = y_core.to(device)

            optimizer.zero_grad()
            pred_contra, pred_core = model(xb)
            loss = loss_fn(pred_contra, y_contra) + weight_core * loss_fn(
                pred_core, y_core
            )
            if smoothness_sigma > 0 and smoothness_weight > 0:
                noise = torch.randn_like(xb) * smoothness_sigma
                xb_pert = xb + noise
                pred_contra_pert, pred_core_pert = model(xb_pert)
                smooth_loss = loss_fn(pred_contra_pert, pred_contra) + weight_core * loss_fn(
                    pred_core_pert, pred_core
                )
                loss = loss + smoothness_weight * smooth_loss

            if physics_weight > 0:
                # Enforce agreement with the slow physical minimizer on a small subset to keep runtime reasonable.
                n_phys = min(physics_max_batch, xb.shape[0])
                params_phys_scaled = xb[:n_phys].cpu().numpy()
                params_phys = scalers["x"].inverse_transform(params_phys_scaled)
                phys_targets = []
                for row in params_phys:
                    try:
                        log_r0, log_mbar, log_m200, log_c200 = row
                        r0 = 10.0 ** log_r0
                        mbar = 10.0 ** log_mbar
                        m200 = 10.0 ** log_m200
                        c200 = 10.0 ** log_c200
                        halo_init = pr.NFW(m200, c200, Delta=200.0, z=0.0)
                        disk = pr.Hernquist(mbar, r0)
                        MvD, cD, aD = gh._contra(R_FULLRANGE, halo_init, disk)
                        phys_targets.append([np.log10(MvD), cD, aD])
                    except Exception:
                        continue
                if phys_targets:
                    phys_targets = np.asarray(phys_targets, dtype=np.float32)
                    phys_scaled = scalers["contra"].transform(phys_targets)
                    idx_len = phys_scaled.shape[0]
                    tgt = torch.from_numpy(phys_scaled).to(device)
                    physics_loss = loss_fn(pred_contra[:idx_len], tgt)
                    loss = loss + physics_weight * physics_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, y_contra, y_core in val_loader:
                xb = xb.to(device)
                y_contra = y_contra.to(device)
                y_core = y_core.to(device)
                pred_contra, pred_core = model(xb)
                loss = loss_fn(pred_contra, y_contra) + weight_core * loss_fn(
                    pred_core, y_core
                )
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        history.append((epoch, train_loss, val_loss))
        if not quiet:
            print(f"Epoch {epoch:03d} | train {train_loss:.4e} | val {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            payload = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_features": features.shape[1],
                    "hidden_dim": hidden_dim,
                    "depth": depth,
                    "dropout": dropout,
                },
                "scalers": {
                    "x": scalers["x"].to_dict(),
                    "contra": scalers["contra"].to_dict(),
                    "core": scalers["core"].to_dict(),
                },
                "sampler_config": sampler.__dict__,
                "training_stats": {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "n_samples": len(features),
                },
            }
            torch.save(payload, output_path)

    return {
        "best_val_loss": best_val,
        "history": history,
        "n_training_samples": len(train_loader.dataset),
        "n_validation_samples": len(val_loader.dataset),
        "model_path": os.path.abspath(output_path),
    }


class HaloSurrogateMLP:
    """Convenience wrapper around a trained residual MLP surrogate."""

    def __init__(self, bundle: Dict[str, dict], device: Optional[str] = None) -> None:
        config = bundle.get("model_config", {})
        self.model = HaloMLP(**config)
        self.model.load_state_dict(bundle["model_state_dict"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        scalers = bundle["scalers"]
        self.x_scaler = Standardizer.from_dict(scalers["x"])
        self.contra_scaler = Standardizer.from_dict(scalers["contra"])
        self.core_scaler = Standardizer.from_dict(scalers["core"])
        self._attach_sampler_priors(bundle.get("sampler_config", {}))

    @classmethod
    def from_file(cls, path: str, device: Optional[str] = None) -> "HaloSurrogateMLP":
        bundle = torch.load(path, map_location="cpu")
        return cls(bundle, device=device)

    def predict(
        self, params: Sequence[Sequence[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.atleast_2d(params).astype(np.float32)
        x = torch.from_numpy(self.x_scaler.transform(arr)).to(self.device)
        with torch.no_grad():
            contra_scaled, core_scaled = self.model(x)
        contra = self.contra_scaler.inverse_transform(contra_scaled.cpu().numpy())
        core = self.core_scaler.inverse_transform(core_scaled.cpu().numpy())
        # Physically-informed clipping without invoking the slow minimizer in gh.contra.
        contra[:, 0] = np.clip(contra[:, 0], self._log_m200_min, self._log_m200_max + 0.5)
        contra[:, 1] = np.clip(contra[:, 1], 0.1, 200.0)
        contra[:, 2] = np.clip(contra[:, 2], 0.01, 50.0)
        core[:, 0] = np.clip(core[:, 0], 0.0, None)
        core[:, 1] = np.clip(core[:, 1], 0.0, None)
        return contra, core

    def _attach_sampler_priors(self, sampler_config: Dict[str, object]) -> None:
        rng = sampler_config.get("log_m200_range", (8.0, 13.0))
        self._log_m200_min = float(rng[0])
        self._log_m200_max = float(rng[1])


def _cli_train(args: argparse.Namespace) -> None:
    sampler = ParameterSampler(
        log_m200_range=tuple(args.log_m200_range),
        log_mbar_min=args.log_mbar_min,
        min_delta_m=args.min_delta_m,
        r0_range_kpc=tuple(args.r0_range),
        c_sigma=args.c_sigma,
        seed=args.seed,
    )
    start = time.time()
    metrics = train_mlp_surrogate(
        n_samples=args.n_samples,
        sampler=sampler,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_fraction=args.val_fraction,
        weight_core=args.weight_core,
        smoothness_sigma=args.smoothness_sigma,
        smoothness_weight=args.smoothness_weight,
        physics_weight=args.physics_weight,
        physics_max_batch=args.physics_max_batch,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        device=args.device,
        output_path=args.output,
        quiet=args.quiet,
    )
    metrics["elapsed_sec"] = time.time() - start
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train halo surrogate (Residual MLP).")
    parser.add_argument("--output", required=True, help="Path for torch model.")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--weight-core", type=float, default=1.0)
    parser.add_argument("--smoothness-sigma", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--physics-weight", type=float, default=0.1,
                        help="Weight for physics-informed loss using gh._contra on a small subset per batch.")
    parser.add_argument("--physics-max-batch", type=int, default=32,
                        help="Max batch size for expensive physics calls per training step.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-m200-range", nargs=2, type=float, default=(8.0, 13.0)
    )
    parser.add_argument("--log-mbar-min", type=float, default=6.5)
    parser.add_argument(
        "--r0-range", nargs=2, type=float, default=(0.05, 50.0)
    )
    parser.add_argument("--c-sigma", type=float, default=0.11)
    parser.add_argument("--min-delta-m", type=float, default=0.15)
    parser.set_defaults(func=_cli_train)
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
