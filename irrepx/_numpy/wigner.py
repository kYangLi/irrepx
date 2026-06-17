"""Wigner D matrices from direction vectors (numpy).

Computes rotation matrices from edge direction vectors via
Gram-Schmidt → Euler angles → JD-seed Wigner D formula.
"""

import functools
from typing import List

import numpy as np

from irrepx._constants import load_jd
from irrepx._constants._compute import jd_seed


@functools.cache
def _jd() -> list[np.ndarray]:
    """Lazily loaded JD seed matrices (cached after first call)."""
    return load_jd()


NORM_BASE_X_AXIS = np.array([0.7562168147812394, 0.6543211207366891, 0.0], dtype=np.float64)


def wigner_D_from_direction(
    edge_vec: np.ndarray, l_range: List[int], *, use_precomputed_jd: bool = True
) -> List[np.ndarray]:
    """Compute Wigner D matrices from edge direction vectors.

    Args:
        edge_vec: (E, 3) unit direction vectors.
        l_range: list of l values, e.g. [0, 1, 2, ..., l_max].
        use_precomputed_jd: use JD matrices from the precomputed npz table (default True).

    Returns:
        list[np.ndarray], one (E, 2l+1, 2l+1) matrix per l.
    """
    edge_rot_mat = _init_edge_rot_mat(edge_vec)
    return _rotation_to_wigner_D_list(edge_rot_mat, l_range, use_precomputed_jd=use_precomputed_jd)


def _init_edge_rot_mat(edge_vec: np.ndarray):
    norm_x = edge_vec / np.linalg.norm(edge_vec, axis=1, keepdims=True)
    edge_vec_2 = np.tile(NORM_BASE_X_AXIS, (edge_vec.shape[0], 1))
    edge_vec_2b = np.stack([-edge_vec_2[:, 1], edge_vec_2[:, 0], edge_vec_2[:, 2]], axis=1)
    edge_vec_2c = np.stack([edge_vec_2[:, 0], -edge_vec_2[:, 2], edge_vec_2[:, 1]], axis=1)
    dots = np.abs(np.einsum("ij,ij->i", edge_vec_2, norm_x))
    dots_b = np.abs(np.einsum("ij,ij->i", edge_vec_2b, norm_x))
    dots_c = np.abs(np.einsum("ij,ij->i", edge_vec_2c, norm_x))
    min_indices = np.argmin(np.stack([dots, dots_b, dots_c], axis=1), axis=1)
    edge_vec_2 = np.take_along_axis(
        np.stack([edge_vec_2, edge_vec_2b, edge_vec_2c], axis=1), min_indices[:, None, None], axis=1
    ).squeeze(1)
    norm_z = np.cross(norm_x, edge_vec_2, axis=1)
    norm_z /= np.linalg.norm(norm_z, axis=1, keepdims=True)
    norm_y = np.cross(norm_x, norm_z, axis=1)
    norm_y /= np.linalg.norm(norm_y, axis=1, keepdims=True)
    return np.transpose(
        np.concatenate([norm_z.reshape(-1, 3, 1), norm_x.reshape(-1, 3, 1), (-norm_y).reshape(-1, 3, 1)], axis=2),
        (0, 2, 1),
    )


def _rotation_to_wigner_D_list(
    edge_rot_mat: np.ndarray, l_range: List, *, use_precomputed_jd: bool = True
) -> List[np.ndarray]:
    x = edge_rot_mat[:, :, 1]
    alpha, beta = _xyz_to_angles(x)
    y = _angles_to_matrix(alpha, beta, np.zeros_like(alpha))
    R = y.mT @ edge_rot_mat
    gamma = np.atan2(R[..., 0, 2], R[..., 0, 0])
    return [_wigner_D(ll, alpha, beta, gamma, use_precomputed_jd=use_precomputed_jd) for ll in l_range]


def _wigner_D(
    lv: int, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, *, use_precomputed_jd: bool = True
) -> np.ndarray:
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    J = _jd()[lv] if use_precomputed_jd else jd_seed(lv)
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: np.ndarray, lv: int) -> np.ndarray:
    shape = angle.shape
    M = np.zeros((*shape, 2 * lv + 1, 2 * lv + 1))
    indexes = np.arange(0, 2 * lv + 1, 1)
    reversed_indexes = np.arange(2 * lv, -1, -1)
    frequencies = np.arange(lv, -lv - 1, -1)
    M[..., indexes, reversed_indexes] = np.sin(frequencies * angle[..., None])
    M[..., indexes, indexes] = np.cos(frequencies * angle[..., None])
    return M


def _xyz_to_angles(xyz):
    xyz = _normalize(xyz)
    xyz = np.clip(xyz, -1, 1)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    x_ = np.where((x == 0.0) & (z == 0.0), 0.0, x)
    y_ = np.where((x == 0.0) & (z == 0.0), 0.0, y)
    z_ = np.where((x == 0.0) & (z == 0.0), 1.0, z)
    beta = np.where(y == 1.0, 0.0, np.where(y == -1, np.pi, np.arccos(y_)))
    alpha = np.arctan2(x_, z_)
    return alpha, beta


def _normalize(x):
    n2 = np.sum(x**2, axis=-1, keepdims=True)
    n2 = np.where(n2 > 0.0, n2, 1.0)
    return x / np.sqrt(n2)


def _angles_to_matrix(alpha, beta, gamma):
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    return _matrix_y(alpha) @ _matrix_x(beta) @ _matrix_y(gamma)


def _matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    o = np.ones_like(angle)
    z = np.zeros_like(angle)
    return np.stack(
        [np.stack([c, z, s], axis=-1), np.stack([z, o, z], axis=-1), np.stack([-s, z, c], axis=-1)],
        axis=-2,
    )


def _matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    o = np.ones_like(angle)
    z = np.zeros_like(angle)
    return np.stack(
        [np.stack([o, z, z], axis=-1), np.stack([z, c, -s], axis=-1), np.stack([z, s, c], axis=-1)],
        axis=-2,
    )
