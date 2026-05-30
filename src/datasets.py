from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset

from src.utils import ensure_dir

AVAILABLE_DATASETS = ("gaussian_mixture", "ring", "two_moons", "spiral")
AVAILABLE_CONTAMINATION_KINDS = ("uniform", "cluster_shift", "heteroscedastic")
DATASET_TO_LABEL = {name: idx for idx, name in enumerate(AVAILABLE_DATASETS)}
LABEL_TO_DATASET = {idx: name for name, idx in DATASET_TO_LABEL.items()}
OFFICIAL_SPLIT_FILES = ("train.npy", "test.npy", "train_label.npy", "test_label.npy")
OFFICIAL_HIDDEN_FILES = ("hidden_test.npy", "hidden_test_label.npy")


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    x_test: np.ndarray
    meta: dict[str, object]


@dataclass
class LabeledDatasetBundle:
    names: tuple[str, ...]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, object]


def _gaussian_mixture(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n_modes = 8
    radius = 2.6
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    mode_ids = rng.integers(0, n_modes, size=n_samples)
    noise = rng.normal(scale=0.18, size=(n_samples, 2))
    return centers[mode_ids] + noise


def _ring(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    radii = rng.normal(loc=2.2, scale=0.12, size=n_samples)
    x = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    x += rng.normal(scale=0.03, size=x.shape)
    return x


def _two_moons(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    x, _ = make_moons(
        n_samples=n_samples,
        noise=0.08,
        random_state=int(rng.integers(1_000_000_000)),
    )
    x = (x - x.mean(axis=0)) * np.array([2.2, 2.2])
    return x


def _spiral(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.3, 4.8 * np.pi, size=n_samples)
    r = 0.18 * t
    x = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x += rng.normal(scale=0.12, size=x.shape)
    x = x / np.max(np.linalg.norm(x, axis=1)) * 3.1
    return x


GENERATORS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "gaussian_mixture": _gaussian_mixture,
    "ring": _ring,
    "two_moons": _two_moons,
    "spiral": _spiral,
}


def _official_dataset_exists(root: str | Path) -> bool:
    root_path = Path(root)
    return all((root_path / filename).exists() for filename in OFFICIAL_SPLIT_FILES)


def _official_hidden_exists(root: str | Path) -> bool:
    root_path = Path(root)
    return all((root_path / filename).exists() for filename in OFFICIAL_HIDDEN_FILES)


def _candidate_npz_paths(name: str, root: str | Path) -> list[Path]:
    root_path = Path(root)
    candidates = [root_path / f"{name}.npz"]
    if root_path.name != "processed":
        candidates.append(root_path / "processed" / f"{name}.npz")
    return candidates


def _default_npz_root(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.name == "processed":
        return root_path
    return root_path / "processed"


def _load_official_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    root_path = Path(root)
    x_train = np.load(root_path / "train.npy").astype(np.float32)
    x_test = np.load(root_path / "test.npy").astype(np.float32)
    y_train = np.load(root_path / "train_label.npy").astype(np.int64)
    y_test = np.load(root_path / "test_label.npy").astype(np.int64)

    metadata_path = root_path / "metadata.json"
    meta: dict[str, object] = {"source": "readme_data_format", "root": str(root_path)}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        meta.update(loaded_meta)
    return x_train, y_train, x_test, y_test, meta


def _load_official_hidden_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    root_path = Path(root)
    x_hidden = np.load(root_path / "hidden_test.npy").astype(np.float32)
    y_hidden = np.load(root_path / "hidden_test_label.npy").astype(np.int64)

    metadata_path = root_path / "metadata.json"
    meta: dict[str, object] = {"source": "readme_data_format", "root": str(root_path), "split": "hidden_test"}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        meta.update(loaded_meta)
    return x_hidden, y_hidden, meta


def _slice_official_dataset(name: str, root: str | Path) -> DatasetBundle:
    x_train, y_train, x_test, y_test, meta = _load_official_arrays(root)
    label = DATASET_TO_LABEL[name]
    bundle_meta = dict(meta)
    bundle_meta.update(
        {
            "name": name,
            "label": label,
            "n_train": int(np.sum(y_train == label)),
            "n_test": int(np.sum(y_test == label)),
            "source": "readme_data_format",
        }
    )
    return DatasetBundle(
        name=name,
        x_train=x_train[y_train == label],
        x_test=x_test[y_test == label],
        meta=bundle_meta,
    )


def load_hidden_dataset(name: str, root: str | Path = "data") -> DatasetBundle:
    if not _official_hidden_exists(root):
        raise FileNotFoundError("Hidden split files are not available under the given data root.")
    x_hidden, y_hidden, meta = _load_official_hidden_arrays(root)
    label = DATASET_TO_LABEL[name]
    bundle_meta = dict(meta)
    bundle_meta.update(
        {
            "name": name,
            "label": label,
            "n_train": int(np.sum(y_hidden == label)),
            "n_test": int(np.sum(y_hidden == label)),
            "source": "readme_hidden_split",
        }
    )
    hidden = x_hidden[y_hidden == label]
    return DatasetBundle(name=name, x_train=hidden, x_test=hidden, meta=bundle_meta)


def generate_dataset(name: str, n_samples: int, seed: int) -> np.ndarray:
    if name not in GENERATORS:
        raise ValueError(f"Unknown dataset: {name}")
    rng = np.random.default_rng(seed)
    return GENERATORS[name](n_samples, rng).astype(np.float32)


def make_dataset_bundle(
    name: str,
    n_train: int = 4000,
    n_test: int = 1000,
    seed: int = 42,
) -> DatasetBundle:
    x_train = generate_dataset(name, n_train, seed)
    x_test = generate_dataset(name, n_test, seed + 10_000)
    meta = {
        "name": name,
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "source": "synthetic_fallback",
    }
    return DatasetBundle(name=name, x_train=x_train, x_test=x_test, meta=meta)


def save_dataset_bundle(bundle: DatasetBundle, root: str | Path = "data/processed") -> Path:
    root = ensure_dir(root)
    path = root / f"{bundle.name}.npz"
    np.savez_compressed(
        path,
        x_train=bundle.x_train,
        x_test=bundle.x_test,
        meta=json.dumps(bundle.meta, ensure_ascii=False),
    )
    return path


def generate_and_save_all(
    root: str | Path = "data/processed",
    n_train: int = 4000,
    n_test: int = 1000,
    seed: int = 42,
) -> list[Path]:
    paths = []
    for offset, name in enumerate(AVAILABLE_DATASETS):
        bundle = make_dataset_bundle(name, n_train=n_train, n_test=n_test, seed=seed + offset * 100)
        paths.append(save_dataset_bundle(bundle, root=root))
    return paths


def load_dataset(name: str, root: str | Path = "data") -> DatasetBundle:
    if _official_dataset_exists(root):
        return _slice_official_dataset(name, root=root)

    path = next((candidate for candidate in _candidate_npz_paths(name, root) if candidate.exists()), None)
    if path is None:
        bundle = make_dataset_bundle(name)
        save_dataset_bundle(bundle, root=_default_npz_root(root))
        return bundle
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    return DatasetBundle(name=name, x_train=data["x_train"], x_test=data["x_test"], meta=meta)


def load_or_generate_dataset(
    name: str,
    root: str | Path = "data",
    n_train: int = 4000,
    n_test: int = 1000,
    seed: int = 42,
) -> DatasetBundle:
    if _official_dataset_exists(root):
        return load_dataset(name, root=root)
    if any(candidate.exists() for candidate in _candidate_npz_paths(name, root)):
        return load_dataset(name, root=root)
    bundle = make_dataset_bundle(name, n_train=n_train, n_test=n_test, seed=seed)
    save_dataset_bundle(bundle, root=_default_npz_root(root))
    return bundle


def make_tensor_dataset(x: np.ndarray) -> TensorDataset:
    tensor = torch.from_numpy(x.astype(np.float32))
    return TensorDataset(tensor)


def make_labeled_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    x_tensor = torch.from_numpy(x.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.int64))
    return TensorDataset(x_tensor, y_tensor)


def load_all_labeled_datasets(
    root: str | Path = "data",
    n_train: int = 4000,
    n_test: int = 1000,
    seed: int = 42,
) -> LabeledDatasetBundle:
    if _official_dataset_exists(root):
        x_train, y_train, x_test, y_test, meta = _load_official_arrays(root)
        return LabeledDatasetBundle(
            names=AVAILABLE_DATASETS,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            meta=meta,
        )

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    meta: dict[str, object] = {
        "names": list(AVAILABLE_DATASETS),
        "n_train_per_dataset": n_train,
        "n_test_per_dataset": n_test,
        "seed": seed,
    }
    for offset, name in enumerate(AVAILABLE_DATASETS):
        bundle = load_or_generate_dataset(
            name,
            root=root,
            n_train=n_train,
            n_test=n_test,
            seed=seed + offset * 100,
        )
        label = DATASET_TO_LABEL[name]
        train_parts.append(bundle.x_train)
        test_parts.append(bundle.x_test)
        train_labels.append(np.full(len(bundle.x_train), label, dtype=np.int64))
        test_labels.append(np.full(len(bundle.x_test), label, dtype=np.int64))
    return LabeledDatasetBundle(
        names=AVAILABLE_DATASETS,
        x_train=np.concatenate(train_parts, axis=0),
        y_train=np.concatenate(train_labels, axis=0),
        x_test=np.concatenate(test_parts, axis=0),
        y_test=np.concatenate(test_labels, axis=0),
        meta=meta,
    )


def load_all_hidden_labeled_datasets(root: str | Path = "data") -> LabeledDatasetBundle:
    if not _official_hidden_exists(root):
        raise FileNotFoundError("Hidden split files are not available under the given data root.")
    x_hidden, y_hidden, meta = _load_official_hidden_arrays(root)
    return LabeledDatasetBundle(
        names=AVAILABLE_DATASETS,
        x_train=x_hidden,
        y_train=y_hidden,
        x_test=x_hidden,
        y_test=y_hidden,
        meta=meta,
    )


def add_outlier_contamination(
    x: np.ndarray,
    ratio: float,
    seed: int = 42,
    low: float = -3.5,
    high: float = 3.5,
) -> np.ndarray:
    if ratio <= 0:
        return x.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    n_outliers = max(1, int(len(x) * ratio))
    outliers = rng.uniform(low=low, high=high, size=(n_outliers, x.shape[1])).astype(np.float32)
    mixed = np.concatenate([x.astype(np.float32), outliers], axis=0)
    rng.shuffle(mixed)
    return mixed


def add_cluster_shift_contamination(
    x: np.ndarray,
    ratio: float,
    seed: int = 42,
    shift_scale: float = 1.35,
) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if ratio <= 0:
        return x
    rng = np.random.default_rng(seed)
    n_points = max(1, int(len(x) * ratio))
    indices = rng.choice(len(x), size=n_points, replace=False)
    shift = rng.normal(loc=0.0, scale=shift_scale, size=(1, x.shape[1])).astype(np.float32)
    x[indices] = x[indices] + shift
    return x


def add_heteroscedastic_contamination(
    x: np.ndarray,
    ratio: float,
    seed: int = 42,
    noise_low: float = 0.25,
    noise_high: float = 1.0,
) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if ratio <= 0:
        return x
    rng = np.random.default_rng(seed)
    n_points = max(1, int(len(x) * ratio))
    indices = rng.choice(len(x), size=n_points, replace=False)
    scales = rng.uniform(noise_low, noise_high, size=(n_points, 1)).astype(np.float32)
    noise = rng.normal(size=(n_points, x.shape[1])).astype(np.float32) * scales
    x[indices] = x[indices] + noise
    return x


def apply_contamination(
    x: np.ndarray,
    ratio: float,
    kind: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    if kind == "uniform":
        return add_outlier_contamination(x, ratio=ratio, seed=seed)
    if kind == "cluster_shift":
        return add_cluster_shift_contamination(x, ratio=ratio, seed=seed)
    if kind == "heteroscedastic":
        return add_heteroscedastic_contamination(x, ratio=ratio, seed=seed)
    raise ValueError(f"Unknown contamination kind: {kind}")


def make_contaminated_bundle(
    bundle: DatasetBundle,
    ratio: float,
    seed: int = 42,
    kind: str = "uniform",
) -> DatasetBundle:
    x_train = apply_contamination(bundle.x_train, ratio=ratio, kind=kind, seed=seed)
    meta = dict(bundle.meta)
    meta["contamination_ratio"] = ratio
    meta["contamination_kind"] = kind
    meta["source"] = f"{meta.get('source', 'synthetic_fallback')}+{kind}_contamination"
    meta["n_train_clean"] = len(bundle.x_train)
    meta["n_train_total"] = len(x_train)
    return DatasetBundle(name=bundle.name, x_train=x_train, x_test=bundle.x_test, meta=meta)


def replace_with_official_dataset(
    name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    root: str | Path = "data/processed",
) -> Path:
    bundle = DatasetBundle(
        name=name,
        x_train=x_train.astype(np.float32),
        x_test=x_test.astype(np.float32),
        meta={"name": name, "source": "official_or_custom"},
    )
    return save_dataset_bundle(bundle, root=root)
