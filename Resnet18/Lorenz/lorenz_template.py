"""Lorenz 2D template generation and KD-tree nearest-neighbor assignment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LorenzParams:
    x0: Tuple[float, float, float] = (0.1, 0.0, 0.0)
    dt: float = 0.001
    n_template: int = 2**16
    skip_transient: int = 10000
    sigma: float = 5.0
    rho: float = 14.0
    beta: float = 1.0
    dtype: str = "float32"


@dataclass
class LorenzTemplate2D:
    template: np.ndarray
    params: LorenzParams
    pca_components: np.ndarray
    mean: np.ndarray
    scale_params: Dict[str, float]
    explained_variance_ratio: np.ndarray


def lorenz_rhs(
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> np.ndarray:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


def integrate_lorenz(params: LorenzParams, verbose: bool = False) -> np.ndarray:
    """Integrate the Lorenz system once and return the full 3D trajectory."""
    dtype = np.dtype(params.dtype)
    n_total = params.skip_transient + params.n_template
    traj = np.zeros((n_total, 3), dtype=dtype)
    state = np.array(params.x0, dtype=float)

    print_interval = max(1, n_total // 100) if verbose and n_total > 10_000 else 0
    start_time = time.time()
    if verbose and n_total > 10_000:
        print(f"  Integrating Lorenz trajectory: {n_total:,} steps...", flush=True)

    for i in range(n_total):
        traj[i] = state

        k1 = lorenz_rhs(state, params.sigma, params.rho, params.beta)
        k2 = lorenz_rhs(state + 0.5 * params.dt * k1, params.sigma, params.rho, params.beta)
        k3 = lorenz_rhs(state + 0.5 * params.dt * k2, params.sigma, params.rho, params.beta)
        k4 = lorenz_rhs(state + params.dt * k3, params.sigma, params.rho, params.beta)
        state = state + (params.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if print_interval and (i + 1) % print_interval == 0:
            progress = (i + 1) / n_total * 100
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_total - i - 1) if i > 0 else 0.0
            print(
                f"  Progress: {progress:.1f}% ({i + 1:,}/{n_total:,}) | "
                f"elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                flush=True,
            )

    if verbose and n_total > 10_000:
        total_time = time.time() - start_time
        print(f"  Lorenz integration done in {total_time:.1f}s", flush=True)

    return traj[params.skip_transient :]


def pca_project_to_2d(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a 3D trajectory onto its top two PCA components."""
    mean = traj.mean(axis=0)
    traj_centered = traj - mean
    cov_matrix = np.cov(traj_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    pca_components = eigenvectors.T
    points_2d = traj_centered @ pca_components[:2].T
    return points_2d, pca_components, explained_variance_ratio, mean


def scale_to_unit_square(points_2d: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Scale 2D points to [0, 1]^2 and return scale parameters."""
    x_min = float(points_2d[:, 0].min())
    x_max = float(points_2d[:, 0].max())
    y_min = float(points_2d[:, 1].min())
    y_max = float(points_2d[:, 1].max())

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    scaled = np.empty_like(points_2d, dtype=np.float32)
    scaled[:, 0] = (points_2d[:, 0] - x_min) / x_range
    scaled[:, 1] = (points_2d[:, 1] - y_min) / y_range

    scale_params = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_range": x_range,
        "y_range": y_range,
    }
    return scaled, scale_params


def build_lorenz_template_2d(params: LorenzParams, verbose: bool = False) -> LorenzTemplate2D:
    """Generate the full Lorenz 2D template in [0, 1]^2."""
    if verbose:
        print("Building Lorenz 2D template...")

    traj = integrate_lorenz(params, verbose=verbose)
    points_2d, pca_components, explained_variance_ratio, mean = pca_project_to_2d(traj)
    template, scale_params = scale_to_unit_square(points_2d)

    if verbose:
        cumulative = explained_variance_ratio[0] + explained_variance_ratio[1]
        print(f"  Template points: {template.shape[0]:,}")
        print(f"  Top-2 PCA variance: {cumulative * 100:.2f}%")

    return LorenzTemplate2D(
        template=template,
        params=params,
        pca_components=pca_components,
        mean=mean,
        scale_params=scale_params,
        explained_variance_ratio=explained_variance_ratio,
    )


def build_kdtree(template: np.ndarray, leafsize: int = 16, verbose: bool = False) -> cKDTree:
    """Build a KD-tree on the full 2D template."""
    if verbose:
        print(f"Building KD-tree on {template.shape[0]:,} template points (leafsize={leafsize})...")
    kdtree = cKDTree(template, leafsize=leafsize)
    if verbose:
        print("  KD-tree ready.")
    return kdtree


def build_lorenz_compressor_2d(
    params: LorenzParams,
    leafsize: int = 16,
    verbose: bool = False,
) -> Tuple[LorenzTemplate2D, cKDTree]:
    """Build the Lorenz 2D template and its KD-tree index."""
    lorenz_template = build_lorenz_template_2d(params=params, verbose=verbose)
    lorenz_tree = build_kdtree(lorenz_template.template, leafsize=leafsize, verbose=verbose)
    return lorenz_template, lorenz_tree


def query_nearest_indices(
    weight_points_2d: np.ndarray,
    kdtree: cKDTree,
    workers: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch nearest-neighbor query for all points to compress."""
    points = np.asarray(weight_points_2d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"weight_points_2d must have shape (n, 2), got {points.shape}")

    distances, indices = kdtree.query(points, workers=workers)
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def reconstruct_from_indices(indices: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Recover 2D points by direct template indexing."""
    idx = np.asarray(indices, dtype=np.int64)
    return template[idx]
