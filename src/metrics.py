from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


def _subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def _median_bandwidth(x: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    z = np.concatenate([x, y], axis=0)
    z = _subsample(z, max_points=1500, seed=seed)
    sq_dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median = float(np.median(tri))
    return max(math.sqrt(median + 1e-8), 1e-3)


def rbf_mmd(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None, seed: int = 42) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bandwidth = bandwidth or _median_bandwidth(x, y, seed=seed)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    xx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    yy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    xy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 128,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(n_projections, x.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    distances = []
    for direction in projections:
        px = x @ direction
        py = y @ direction
        distances.append(wasserstein_distance(px, py))
    return float(np.mean(distances))


def chamfer_distance(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    max_points: int = 1000,
    seed: int = 42,
) -> float:
    x_real = _subsample(np.asarray(x_real, dtype=np.float64), max_points=max_points, seed=seed)
    x_generated = _subsample(np.asarray(x_generated, dtype=np.float64), max_points=max_points, seed=seed + 1)
    distances = cdist(x_real, x_generated, metric="euclidean")
    forward = distances.min(axis=1).mean()
    backward = distances.min(axis=0).mean()
    return float(0.5 * (forward + backward))


def kde_nll(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    bandwidth: float | None = None,
    seed: int = 42,
) -> float:
    x_real = np.asarray(x_real, dtype=np.float64)
    x_generated = np.asarray(x_generated, dtype=np.float64)
    bandwidth = bandwidth or _median_bandwidth(x_real, x_generated, seed=seed)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    fit_samples = _subsample(x_generated, max_points=2000, seed=seed)
    eval_samples = _subsample(x_real, max_points=2000, seed=seed + 1)
    kde.fit(fit_samples)
    return float(-kde.score_samples(eval_samples).mean())


def gmm_nll(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    max_components: int = 8,
    seed: int = 42,
) -> float:
    x_real = np.asarray(x_real, dtype=np.float64)
    x_generated = np.asarray(x_generated, dtype=np.float64)
    fit_samples = _subsample(x_generated, max_points=3000, seed=seed)
    eval_samples = _subsample(x_real, max_points=3000, seed=seed + 1)
    n_components = max(1, min(max_components, int(np.sqrt(len(fit_samples)) // 8), len(fit_samples)))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        random_state=seed,
    )
    gmm.fit(fit_samples)
    return float(-gmm.score_samples(eval_samples).mean())


def mode_coverage_score(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    n_modes: int = 8,
    threshold_scale: float = 1.8,
    seed: int = 42,
) -> float:
    n_modes = min(n_modes, len(x_real))
    kmeans = KMeans(n_clusters=n_modes, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(x_real)
    covered = 0
    for idx in range(n_modes):
        cluster_points = x_real[labels == idx]
        center = kmeans.cluster_centers_[idx]
        if len(cluster_points) == 0:
            continue
        radius = np.mean(np.linalg.norm(cluster_points - center, axis=1)) * threshold_scale + 1e-6
        min_dist = np.min(np.linalg.norm(x_generated - center, axis=1))
        covered += float(min_dist <= radius)
    return float(covered / n_modes)


def summarize_metrics(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    return {
        "mmd": rbf_mmd(x_real, x_generated, seed=seed),
        "sliced_wasserstein": sliced_wasserstein_distance(x_real, x_generated, seed=seed),
        "chamfer": chamfer_distance(x_real, x_generated, seed=seed),
        "kde_nll": kde_nll(x_real, x_generated, seed=seed),
        "gmm_nll": gmm_nll(x_real, x_generated, seed=seed),
        "mode_coverage": mode_coverage_score(x_real, x_generated, seed=seed),
    }
