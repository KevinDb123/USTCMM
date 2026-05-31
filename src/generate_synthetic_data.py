from __future__ import annotations

import argparse

from src.datasets import AVAILABLE_DATASETS, generate_and_save_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic train/test datasets for 2D generative modeling.")
    parser.add_argument("--root", type=str, default="data/processed")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = generate_and_save_all(root=args.root, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    print("Generated datasets:")
    for name, path in zip(AVAILABLE_DATASETS, paths):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
