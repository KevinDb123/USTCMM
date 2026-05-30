from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.datasets import AVAILABLE_DATASETS, load_or_generate_dataset
from src.utils import ensure_dir


def summarize_array(x: np.ndarray) -> dict[str, float]:
    radii = np.linalg.norm(x, axis=1)
    cov = np.cov(x.T)
    return {
        "mean_x": float(np.mean(x[:, 0])),
        "mean_y": float(np.mean(x[:, 1])),
        "std_x": float(np.std(x[:, 0])),
        "std_y": float(np.std(x[:, 1])),
        "min_x": float(np.min(x[:, 0])),
        "max_x": float(np.max(x[:, 0])),
        "min_y": float(np.min(x[:, 1])),
        "max_y": float(np.max(x[:, 1])),
        "mean_radius": float(np.mean(radii)),
        "std_radius": float(np.std(radii)),
        "cov_trace": float(np.trace(cov)),
    }


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for offset, name in enumerate(AVAILABLE_DATASETS):
        bundle = load_or_generate_dataset(
            name,
            root=args.data_root,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed + offset * 100,
        )
        row = {"dataset": name}
        row.update({f"train_{k}": v for k, v in summarize_array(bundle.x_train).items()})
        row.update({f"test_{k}": v for k, v in summarize_array(bundle.x_test).items()})
        rows.append(row)
    path = Path(args.output_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved dataset summary to {path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute descriptive statistics for all synthetic datasets.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-path", type=str, default="outputs/tables/optional/dataset_summary.csv")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
