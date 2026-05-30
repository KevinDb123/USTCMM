from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-model JSON metrics into a CSV table.")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--output", type=str, default="outputs/tables/summary.csv")
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    rows = []
    for path in sorted(table_dir.glob("*_metrics.json")):
        stem = path.stem.replace("_metrics", "")
        model, dataset = stem.split("_", 1)
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"model": model, "dataset": dataset}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "model"]).reset_index(drop=True)
    ensure_dir(Path(args.output).parent)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
