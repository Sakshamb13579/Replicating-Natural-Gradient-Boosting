#!/usr/bin/env python3
"""Deterministic heteroscedastic regression generator — NGBoost constructed v1.

Formulas (frozen by handoff 007-NG-CD1):
  x1, x2 ~ Uniform(-2, 2) independently
  mu(x) = sin(x1) + x2^2
  z ~ Normal(0, 1)
  regime A: sigma(x) = 0.1 + 0.5 * abs(x1)
  regime B: sigma(x) = 0.6
  regime C: sigma(x) = 2 * (0.1 + 0.5 * abs(x1))
  y = mu(x) + sigma(x) * z

Base rows: n=4000, data_rng seed=0.
Split: 60/20/20 train/validation/test, split_rng seed=1, same assignment all regimes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

N_BASE = 4000
DATA_SEED = 0
SPLIT_SEED = 1
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
REGIMES = ("A", "B", "C")
SPLIT_LABELS = ("train", "validation", "test")


def sigma_for_regime(regime: str, abs_x1: np.ndarray) -> np.ndarray:
    hetero = 0.1 + 0.5 * abs_x1
    if regime == "A":
        return hetero
    if regime == "B":
        return np.full_like(abs_x1, 0.6, dtype=np.float64)
    if regime == "C":
        return 2.0 * hetero
    raise ValueError(f"unknown regime: {regime}")


def assign_splits(n: int, fractions: tuple[float, float, float], rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    # Remainder goes to test so counts sum exactly to n.
    n_test = n - n_train - n_val
    labels = np.empty(n, dtype=object)
    labels[perm[:n_train]] = SPLIT_LABELS[0]
    labels[perm[n_train : n_train + n_val]] = SPLIT_LABELS[1]
    labels[perm[n_train + n_val :]] = SPLIT_LABELS[2]
    assert n_train + n_val + n_test == n
    return labels


def generate_base(data_rng: np.random.Generator, split_rng: np.random.Generator) -> dict[str, np.ndarray]:
    x1 = data_rng.uniform(-2.0, 2.0, size=N_BASE)
    x2 = data_rng.uniform(-2.0, 2.0, size=N_BASE)
    noise_z = data_rng.standard_normal(N_BASE)
    true_mu = np.sin(x1) + x2**2
    split = assign_splits(N_BASE, SPLIT_FRACTIONS, split_rng)
    row_id = np.arange(N_BASE, dtype=np.int64)
    return {
        "row_id": row_id,
        "x1": x1,
        "x2": x2,
        "noise_z": noise_z,
        "true_mu": true_mu,
        "split": split,
    }


def expand_regimes(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    abs_x1 = np.abs(base["x1"])
    n_reg = N_BASE * len(REGIMES)
    out: dict[str, np.ndarray] = {
        "row_id": np.tile(base["row_id"], len(REGIMES)),
        "regime": np.repeat(np.array(REGIMES, dtype=object), N_BASE),
        "split": np.tile(base["split"], len(REGIMES)),
        "x1": np.tile(base["x1"], len(REGIMES)),
        "x2": np.tile(base["x2"], len(REGIMES)),
        "true_mu": np.tile(base["true_mu"], len(REGIMES)),
        "noise_z": np.tile(base["noise_z"], len(REGIMES)),
    }
    true_sigma = np.empty(n_reg, dtype=np.float64)
    offset = 0
    for regime in REGIMES:
        true_sigma[offset : offset + N_BASE] = sigma_for_regime(regime, abs_x1)
        offset += N_BASE
    out["true_sigma"] = true_sigma
    out["y"] = out["true_mu"] + out["true_sigma"] * out["noise_z"]
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, columns: list[str], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(arrays[columns[0]])
    with path.open("w", encoding="ascii", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n):
            row = []
            for col in columns:
                val = arrays[col][i]
                if col == "row_id":
                    row.append(int(val))
                elif col in ("regime", "split"):
                    row.append(str(val))
                else:
                    row.append(f"{float(val):.17g}")
            writer.writerow(row)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "processed" / "heteroscedastic-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_rng = np.random.default_rng(DATA_SEED)
    split_rng = np.random.default_rng(SPLIT_SEED)
    base = generate_base(data_rng, split_rng)
    full = expand_regimes(base)

    base_cols = ["row_id", "x1", "x2", "noise_z", "true_mu", "split"]
    full_cols = [
        "row_id",
        "regime",
        "split",
        "x1",
        "x2",
        "true_mu",
        "true_sigma",
        "noise_z",
        "y",
    ]

    base_path = out_dir / "base-rows-v1.csv"
    full_path = out_dir / "heteroscedastic-v1.csv"
    write_csv(base_path, base_cols, base)
    write_csv(full_path, full_cols, full)

    generator_path = Path(__file__).resolve()
    generator_hash = sha256_file(generator_path)
    paper_selection_root = repo_root.parent

    meta = {
        "generator_script": str(generator_path.relative_to(paper_selection_root)),
        "generator_sha256": generator_hash,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "rng_algorithm": type(data_rng.bit_generator).__name__,
        "data_seed": DATA_SEED,
        "split_seed": SPLIT_SEED,
        "n_base": N_BASE,
        "n_regime_rows": N_BASE * len(REGIMES),
        "files": {
            "base-rows-v1.csv": {"sha256": sha256_file(base_path), "rows": N_BASE},
            "heteroscedastic-v1.csv": {
                "sha256": sha256_file(full_path),
                "rows": N_BASE * len(REGIMES),
            },
        },
    }
    meta_path = out_dir / "generation-meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
