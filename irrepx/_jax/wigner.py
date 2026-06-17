"""Wigner D matrices from direction vectors (JAX, JIT-compatible).

Computes rotation matrices from edge direction vectors via
Gram-Schmidt → Euler angles → JD-seed Wigner D formula.
"""

import functools
from typing import List

import jax.numpy as jnp

from irrepx._constants import load_jd
from irrepx._constants._compute import jd_seed


@functools.cache
def _jd() -> list:
    """Lazily loaded JD seed matrices (cached after first call)."""
    return load_jd()


NORM_BASE_X_AXIS = jnp.array([0.7562168147812394, 0.6543211207366891, 0.0], dtype=jnp.float32)


def wigner_D_from_direction(
    edge_vec: jnp.ndarray, l_range: List[int], *, use_precomputed_jd: bool = True
) -> List[jnp.ndarray]:
    """Compute Wigner D matrices from edge direction vectors.

    Args:
        edge_vec: (E, 3) unit direction vectors.
        l_range: list of l values, e.g. [0, 1, 2, ..., l_max].
        use_precomputed_jd: use JD matrices from the precomputed npz table (default True).

    Returns:
        list[jnp.ndarray], one (E, 2l+1, 2l+1) matrix per l.
    """
    edge_rot_mat = _init_edge_rot_mat_jax(edge_vec)
    return _rotation_to_wigner_D_list_jax(edge_rot_mat, l_range, use_precomputed_jd=use_precomputed_jd)


def _init_edge_rot_mat_jax(edge_vec: jnp.ndarray):
    norm_x = edge_vec / (jnp.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-8)
    edge_vec_2 = jnp.tile(NORM_BASE_X_AXIS, (edge_vec.shape[0], 1))
    edge_vec_2b = jnp.stack([-edge_vec_2[:, 1], edge_vec_2[:, 0], edge_vec_2[:, 2]], axis=1)
    edge_vec_2c = jnp.stack([edge_vec_2[:, 0], -edge_vec_2[:, 2], edge_vec_2[:, 1]], axis=1)
    dots = jnp.abs(jnp.einsum("ij,ij->i", edge_vec_2, norm_x))
    dots_b = jnp.abs(jnp.einsum("ij,ij->i", edge_vec_2b, norm_x))
    dots_c = jnp.abs(jnp.einsum("ij,ij->i", edge_vec_2c, norm_x))
    stacked_dots = jnp.stack([dots, dots_b, dots_c], axis=1)
    min_indices = jnp.argmin(stacked_dots, axis=1)
    stacked_vecs = jnp.stack([edge_vec_2, edge_vec_2b, edge_vec_2c], axis=1)
    edge_vec_2 = jnp.take_along_axis(stacked_vecs, min_indices[:, None, None], axis=1).squeeze(1)
    norm_z = jnp.cross(norm_x, edge_vec_2, axis=1)
    norm_z = norm_z / (jnp.linalg.norm(norm_z, axis=1, keepdims=True) + 1e-8)
    norm_y = jnp.cross(norm_x, norm_z, axis=1)
    norm_y = norm_y / (jnp.linalg.norm(norm_y, axis=1, keepdims=True) + 1e-8)
    mat = jnp.stack([norm_z, norm_x, -norm_y], axis=2)
    return jnp.transpose(mat, (0, 2, 1))


def _rotation_to_wigner_D_list_jax(
    edge_rot_mat: jnp.ndarray, l_range: List, *, use_precomputed_jd: bool = True
) -> List[jnp.ndarray]:
    x = edge_rot_mat[:, :, 1]
    alpha, beta = _xyz_to_angles_jax(x)
    y_mat = _angles_to_matrix_jax(alpha, beta, jnp.zeros_like(alpha))
    R = jnp.matmul(jnp.swapaxes(y_mat, -1, -2), edge_rot_mat)
    gamma = jnp.arctan2(R[..., 0, 2], R[..., 0, 0])
    return [_wigner_D_jax(ll, alpha, beta, gamma, use_precomputed_jd=use_precomputed_jd) for ll in l_range]


def _wigner_D_jax(
    lv: int, alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray, *, use_precomputed_jd: bool = True
) -> jnp.ndarray:
    J_np = _jd()[lv] if use_precomputed_jd else jd_seed(lv)
    J = jnp.array(J_np, dtype=alpha.dtype)
    Xa = _z_rot_mat_jax(alpha, lv)
    Xb = _z_rot_mat_jax(beta, lv)
    Xc = _z_rot_mat_jax(gamma, lv)
    res = jnp.matmul(Xa, J)
    res = jnp.matmul(res, Xb)
    res = jnp.matmul(res, J)
    res = jnp.matmul(res, Xc)
    return res


def _z_rot_mat_jax(angle: jnp.ndarray, lv: int) -> jnp.ndarray:
    batch_size = angle.shape[0]
    dim = 2 * lv + 1
    indexes = jnp.arange(0, dim)
    reversed_indexes = jnp.arange(dim - 1, -1, -1)
    frequencies = jnp.arange(lv, -lv - 1, -1)
    args = frequencies[None, :] * angle[:, None]
    sins = jnp.sin(args)
    coss = jnp.cos(args)
    M = jnp.zeros((batch_size, dim, dim), dtype=angle.dtype)
    batch_idx = jnp.arange(batch_size)[:, None]
    idx_range = indexes[None, :]
    rev_idx_range = reversed_indexes[None, :]
    M = M.at[batch_idx, idx_range, rev_idx_range].set(sins)
    M = M.at[batch_idx, idx_range, idx_range].set(coss)
    return M


def _xyz_to_angles_jax(xyz):
    xyz = _normalize_jax(xyz)
    xyz = jnp.clip(xyz, -1.0, 1.0)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    mask = (jnp.abs(x) < 1e-6) & (jnp.abs(z) < 1e-6)
    x_ = jnp.where(mask, 0.0, x)
    z_ = jnp.where(mask, 1.0, z)
    eps = 1e-7
    y_safe = jnp.clip(y, -1.0 + eps, 1.0 - eps)
    beta = jnp.arccos(y_safe)
    alpha = jnp.arctan2(x_, z_)
    return alpha, beta


def _normalize_jax(x):
    n2 = jnp.sum(x**2, axis=-1, keepdims=True)
    return x / jnp.sqrt(jnp.maximum(n2, 1e-12))


def _angles_to_matrix_jax(alpha, beta, gamma):
    return jnp.matmul(jnp.matmul(_matrix_y_jax(alpha), _matrix_x_jax(beta)), _matrix_y_jax(gamma))


def _matrix_y_jax(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    m0 = jnp.stack([c, z, s], axis=-1)
    m1 = jnp.stack([z, o, z], axis=-1)
    m2 = jnp.stack([-s, z, c], axis=-1)
    return jnp.stack([m0, m1, m2], axis=-2)


def _matrix_x_jax(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    m0 = jnp.stack([o, z, z], axis=-1)
    m1 = jnp.stack([z, c, -s], axis=-1)
    m2 = jnp.stack([z, s, c], axis=-1)
    return jnp.stack([m0, m1, m2], axis=-2)
