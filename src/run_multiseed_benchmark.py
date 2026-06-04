"""Multi-seed benchmark: VAE vs Diffusion vs RealNVP Flow.

Runs all three generative models on all four 2D distributions with
multiple random seeds and computes mean ± std for each metric.
Outputs a comprehensive comparison table and individual results.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets import AVAILABLE_DATASETS
from src.train_diffusion import train as train_diffusion
from src.train_flow import train as train_flow
from src.train_vae import train as train_vae
from src.utils import ensure_dir

# ============================================================
# Configuration
# ============================================================

# 5 random seeds for statistical significance
SEEDS = [42, 123, 456, 789, 1024]

# Model configurations (matching existing benchmarks)
VAE_CONFIG = {
    "epochs": 150,
    "batch_size": 256,
    "latent_dim": 4,
    "hidden_dim": 128,
    "depth": 3,
    "dropout": 0.0,
    "lr": 1e-3,
    "beta": 0.5,
    "log_every": 50,
}

DIFFUSION_CONFIG = {
    "epochs": 250,
    "batch_size": 256,
    "hidden_dim": 128,
    "depth": 3,
    "dropout": 0.0,
    "timesteps": 80,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "schedule_type": "linear",
    "time_hidden_dim": 0,
    "lr": 1e-3,
    "log_every": 50,
    "sampler": "ddpm",
}

FLOW_CONFIG = {
    "epochs": 300,
    "batch_size": 256,
    "hidden_dim": 128,
    "depth": 3,
    "num_layers": 8,
    "dropout": 0.0,
    "lr": 1e-3,
    "log_every": 50,
}

METRIC_NAMES = [
    "mmd",
    "sliced_wasserstein",
    "chamfer",
    "kde_nll",
    "gmm_nll",
    "mode_coverage",
    "train_time_sec",
    "sample_time_sec",
]


def run_single_experiment(
    model_type: str,
    dataset: str,
    seed: int,
    config: dict,
    output_dirs: dict[str, str],
    data_root: str,
) -> dict[str, float]:
    """Run one (model, dataset, seed) experiment and return metrics."""
    tag = f"multiseed_s{seed}"
    common = argparse.Namespace(
        dataset=dataset,
        data_root=data_root,
        figure_dir=output_dirs["figures"],
        checkpoint_dir=output_dirs["checkpoints"],
        table_dir=output_dirs["tables"],
        n_train=4000,
        n_test=1000,
        seed=seed,
        device=None,  # auto-detect GPU
        contamination_ratio=0.0,
        contamination_kind="uniform",
        output_tag=tag,
    )

    if model_type == "vae":
        args = argparse.Namespace(
            **{**vars(common), **VAE_CONFIG},
        )
        metrics = train_vae(args)
    elif model_type == "diffusion":
        args = argparse.Namespace(
            **{**vars(common), **DIFFUSION_CONFIG},
            snapshot_steps=[79, 59, 39, 19],
            sampling_steps=None,
        )
        metrics = train_diffusion(args)
    elif model_type == "flow":
        args = argparse.Namespace(
            **{**vars(common), **FLOW_CONFIG},
        )
        metrics = train_flow(args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return metrics


def aggregate_results(
    all_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate results across seeds: compute mean and std for each metric."""
    # Group by (model_type, dataset)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["model_type"], r["dataset"])
        grouped[key].append(r["metrics"])

    summary: dict[str, dict[str, float]] = {}
    for (model, dataset), metrics_list in grouped.items():
        entry: dict[str, float] = {
            "model": model,
            "dataset": dataset,
            "n_seeds": len(metrics_list),
        }
        for metric in METRIC_NAMES:
            values = [m.get(metric, float("nan")) for m in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            if values:
                entry[f"{metric}_mean"] = float(np.mean(values))
                entry[f"{metric}_std"] = float(np.std(values, ddof=1))
            else:
                entry[f"{metric}_mean"] = float("nan")
                entry[f"{metric}_std"] = float("nan")
        summary[f"{model}_{dataset}"] = entry
    return summary


def print_summary_table(summary: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table."""
    models = ["vae", "diffusion", "flow"]
    print("\n" + "=" * 120)
    print("Multi-Seed Benchmark Summary (mean ± std over 5 seeds)")
    print("=" * 120)

    for dataset in AVAILABLE_DATASETS:
        print(f"\n{'─' * 80}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 80}")
        header = (
            f"{'Model':<12} {'MMD':<22} {'SWD':<22} {'Chamfer':<22} "
            f"{'Mode Cov':<12} {'Sample(s)':<14}"
        )
        print(header)
        print("-" * 104)

        for model in models:
            key = f"{model}_{dataset}"
            if key not in summary:
                continue
            s = summary[key]
            mmd_str = f"{s['mmd_mean']:.5f}±{s['mmd_std']:.5f}"
            swd_str = f"{s['sliced_wasserstein_mean']:.4f}±{s['sliced_wasserstein_std']:.4f}"
            chamfer_str = f"{s['chamfer_mean']:.4f}±{s['chamfer_std']:.4f}"
            mc_str = f"{s['mode_coverage_mean']:.2f}±{s['mode_coverage_std']:.2f}"
            time_str = f"{s['sample_time_sec_mean']:.4f}±{s['sample_time_sec_std']:.4f}"

            print(
                f"{model:<12} {mmd_str:<22} {swd_str:<22} {chamfer_str:<22} "
                f"{mc_str:<12} {time_str:<14}"
            )

    print(f"\n{'─' * 80}")
    print("  Average across all datasets")
    print(f"{'─' * 80}")
    header = (
        f"{'Model':<12} {'MMD':<22} {'SWD':<22} {'Chamfer':<22} "
        f"{'Mode Cov':<12} {'Sample(s)':<14}"
    )
    print(header)
    print("-" * 104)

    for model in models:
        avg_mmd_mean = np.mean(
            [summary[f"{model}_{d}"]["mmd_mean"] for d in AVAILABLE_DATASETS if f"{model}_{d}" in summary]
        )
        avg_swd_mean = np.mean(
            [summary[f"{model}_{d}"]["sliced_wasserstein_mean"] for d in AVAILABLE_DATASETS if f"{model}_{d}" in summary]
        )
        avg_cf_mean = np.mean(
            [summary[f"{model}_{d}"]["chamfer_mean"] for d in AVAILABLE_DATASETS if f"{model}_{d}" in summary]
        )
        avg_mc_mean = np.mean(
            [summary[f"{model}_{d}"]["mode_coverage_mean"] for d in AVAILABLE_DATASETS if f"{model}_{d}" in summary]
        )
        avg_time = np.mean(
            [summary[f"{model}_{d}"]["sample_time_sec_mean"] for d in AVAILABLE_DATASETS if f"{model}_{d}" in summary]
        )
        print(
            f"{model:<12} {avg_mmd_mean:<22.5f} {avg_swd_mean:<22.4f} {avg_cf_mean:<22.4f} "
            f"{avg_mc_mean:<12.2f} {avg_time:<14.4f}"
        )

    print("\n" + "=" * 120)


def save_summary_csv(summary: dict[str, dict[str, float]], path: str) -> None:
    """Save summary results to CSV."""
    ensure_dir(Path(path).parent)
    fieldnames = ["model", "dataset", "n_seeds"] + [
        f"{m}_{s}" for m in METRIC_NAMES for s in ["mean", "std"]
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in summary.values():
            writer.writerow(entry)
    print(f"Saved summary CSV to {path}")


def save_detailed_results(all_results: list[dict], path: str) -> None:
    """Save per-seed detailed results."""
    ensure_dir(Path(path).parent)
    serializable = []
    for r in all_results:
        serializable.append(
            {
                "model_type": r["model_type"],
                "dataset": r["dataset"],
                "seed": r["seed"],
                "metrics": {k: (float(v) if not isinstance(v, str) else v) for k, v in r["metrics"].items()},
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed benchmark: VAE vs Diffusion vs Flow"
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--figure-dir", type=str, default="outputs/figures/multiseed"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="outputs/checkpoints/multiseed"
    )
    parser.add_argument(
        "--table-dir", type=str, default="outputs/tables/multiseed"
    )
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--models", type=str, nargs="+", default=["vae", "diffusion", "flow"],
        choices=["vae", "diffusion", "flow"],
        help="Which models to run",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        choices=AVAILABLE_DATASETS,
        help="Which datasets to run (default: all 4)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Random seeds to use",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: auto-detect GPU)",
    )
    args = parser.parse_args()

    datasets = args.datasets or list(AVAILABLE_DATASETS)
    output_dirs = {
        "figures": args.figure_dir,
        "checkpoints": args.checkpoint_dir,
        "tables": args.table_dir,
    }
    for d in output_dirs.values():
        ensure_dir(d)

    total_runs = len(args.models) * len(datasets) * len(args.seeds)
    print(f"Starting multi-seed benchmark:")
    print(f"  Models: {args.models}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"  GPU: {'Available' if __import__('torch').cuda.is_available() else 'NOT AVAILABLE'}")
    print()

    all_results: list[dict[str, Any]] = []
    run_idx = 0
    total_start = time.perf_counter()

    for dataset in datasets:
        for model_type in args.models:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n{'#' * 60}")
                print(f"Run {run_idx}/{total_runs}: {model_type} on {dataset} (seed={seed})")
                print(f"{'#' * 60}")

                run_start = time.perf_counter()
                try:
                    metrics = run_single_experiment(
                        model_type=model_type,
                        dataset=dataset,
                        seed=seed,
                        config={},
                        output_dirs=output_dirs,
                        data_root=args.data_root,
                    )
                    elapsed = time.perf_counter() - run_start
                    all_results.append(
                        {
                            "model_type": model_type,
                            "dataset": dataset,
                            "seed": seed,
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"  ✓ Completed in {elapsed:.1f}s | "
                        f"MMD={metrics.get('mmd', 'N/A'):.5f} | "
                        f"Sample={metrics.get('sample_time_sec', 'N/A'):.4f}s"
                    )
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(f"Benchmark completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Successful: {len(all_results)}/{total_runs}")

    if all_results:
        # Aggregate
        summary = aggregate_results(all_results)

        # Print table
        print_summary_table(summary)

        # Save outputs
        save_summary_csv(summary, f"{args.table_dir}/multiseed_summary.csv")
        save_detailed_results(all_results, f"{args.table_dir}/multiseed_detailed.json")

        # Also save per-dataset summaries
        for dataset in datasets:
            dataset_results = [r for r in all_results if r["dataset"] == dataset]
            if dataset_results:
                save_detailed_results(
                    dataset_results,
                    f"{args.table_dir}/multiseed_{dataset}.json",
                )
    else:
        print("No successful runs to aggregate.")


if __name__ == "__main__":
    main()
