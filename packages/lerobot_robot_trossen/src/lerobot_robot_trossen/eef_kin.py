"""WidowX AI arm kinematics for the relative end-effector action space (numpy only).

Why: the 11-stage ACT goes to the demo joint pose whatever the camera shows (lerobot_trossen
docs/najy_attempts_insights_1014.md insight 1). State-free policies in the *relative EEF*
action space generalize spatially where absolute or relative-joint actions do not
(arXiv 2509.18644, Table III). This module turns a joint chunk into EEF displacements
relative to the chunk's start pose and back.

Convention (one function pair, used by training, offline tools and the robot adapter):
    T0 = FK(q_obs)                      -- measured joints when the chunk is queried
    rel_k = [R0^T (p_k - p0), rotvec(R0^T R_k)]   with (R_k, p_k) = FK(a_k)
i.e. displacement in the chunk-start end-effector frame (the wrist-camera frame moves
with it). Grippers and base velocities stay absolute; only the 6 arm joints per arm change.

Geometry: ``assets/robots/wxai/wxai_follower.xml`` (bodies carry translations only, joint
axes below), ``ee_site`` 0.156 m along link_6 x. Checked against mujoco ``site_xpos`` to
1e-7 (10/16). Real-arm zero offsets match by indirect evidence only -- t02 grasp poses
cluster to 1~5 cm in FK space while their joints spread 0.1~0.33 rad [추정 until the robot
logs ``get_cartesian_positions``].

Must stay importable without mujoco/scipy/torch: the robot PC venv has none of them. The
same file is vendored byte-identical into lerobot_trossen (``rel_eef_patch.py`` uses it;
tests/test_rel_eef_patch.py compares the two when both checkouts are present).
"""

from __future__ import annotations

import numpy as np

# (translation from parent body, joint axis) for joint_0..joint_5 -- wxai_follower.xml
CHAIN = (
    ((0.0, 0.0, 0.05725), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04625), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.245, 0.0, 0.06), (0.0, -1.0, 0.0)),
    ((0.06775, 0.0, 0.0455), (0.0, 0.0, -1.0)),
    ((0.02895, 0.0, -0.0455), (1.0, 0.0, 0.0)),
)
EE_OFFSET = np.array([0.156, 0.0, 0.0])
JOINT_LIMITS = np.array([
    [-3.05433, 3.05433], [0.0, 3.14159], [0.0, 2.35619],
    [-1.5708, 1.5708], [-1.5708, 1.5708], [-3.14159, 3.14159],
])
# The recorded data runs past these by up to 0.063 rad (j4 1.5837; ≤0.04 % of frames, coder 10/16),
# so IK clamps to the limits widened by this much -- clamping at the XML value turned a 0 mm
# round trip into 7 mm / 0.16 rad. The arm's own controller still enforces its real limits.
IK_LIMIT_MARGIN = 0.1
# 16D action layout [left7, right7, x.vel, theta.vel]; arm joints of each arm
ARM_SLICES = ((0, 6), (7, 13))

_T = np.array([c[0] for c in CHAIN])          # (6, 3)
_AX = np.array([c[1] for c in CHAIN])         # (6, 3)


def _skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3)."""
    z = np.zeros(v.shape[:-1])
    return np.stack([
        np.stack([z, -v[..., 2], v[..., 1]], -1),
        np.stack([v[..., 2], z, -v[..., 0]], -1),
        np.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues; axis (3,) or (..., 3) unit, angle (...) -> (..., 3, 3)."""
    angle = np.asarray(angle, dtype=np.float64)
    k = _skew(np.broadcast_to(axis, angle.shape + (3,)))
    s = np.sin(angle)[..., None, None]
    c = (1.0 - np.cos(angle))[..., None, None]
    return np.eye(3) + s * k + c * (k @ k)


def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint angles (..., 6) -> (R (..., 3, 3), p (..., 3)) of ee_site in the arm base frame."""
    q = np.asarray(q, dtype=np.float64)
    R = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
    p = np.zeros(q.shape[:-1] + (3,))
    for i in range(6):
        p = p + np.einsum("...ij,j->...i", R, _T[i])
        R = R @ axis_angle_matrix(_AX[i], q[..., i])
    return R, p + np.einsum("...ij,j->...i", R, EE_OFFSET)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3). Exact for angles < pi. Chunk-relative rotations in the 27 training
    sets stay <= 140.5 deg except one GIST task2 wrist glitch (ep1325, 178.2 deg; adversary 10/17)."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(ang)
    small = s < 1e-8
    scale = np.where(small, 0.5, ang / (2.0 * np.where(small, 1.0, s)))
    return w * scale[..., None]


def matrix_from_rotvec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    ang = np.linalg.norm(v, axis=-1)
    axis = v / np.where(ang < 1e-12, 1.0, ang)[..., None]
    return axis_angle_matrix(axis, ang)


def relative(q0: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """q0 (..., 6) chunk-start joints, qk (..., K, 6) targets -> (..., K, 6) [dp, rotvec]."""
    R0, p0 = fk(q0)
    Rk, pk = fk(qk)
    R0t = np.swapaxes(R0, -1, -2)[..., None, :, :]
    dp = np.einsum("...ij,...j->...i", R0t, pk - p0[..., None, :])
    return np.concatenate([dp, rotvec_from_matrix(R0t @ Rk)], -1)


def absolute_pose(q0: np.ndarray, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``relative`` up to IK: q0 (6,), rel (K, 6) -> target (R (K,3,3), p (K,3))."""
    R0, p0 = fk(q0)
    p = p0 + rel[:, :3] @ R0.T
    R = R0 @ matrix_from_rotvec(rel[:, 3:])
    return R, p


# Every joint axis is a signed coordinate axis, so one joint's rotation is an elementary
# rotation -- the IK path below uses that to stay at ~0.1 ms per FK+Jacobian call.
_AXIS_ID = tuple(int(np.argmax(np.abs(a))) for a in _AX)
_AXIS_SIGN = tuple(float(np.sign(a[i])) for a, i in zip(_AX, _AXIS_ID))


def _elementary(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One joint vector (6,) -> (R (3,3), p (3,), J (6,6) rows [v; w] in the base frame)."""
    R = np.eye(3)
    p = np.zeros(3)
    origins = np.zeros((6, 3))
    axes = np.zeros((6, 3))
    for i in range(6):
        p = p + R @ _T[i]
        axes[i] = R[:, _AXIS_ID[i]] * _AXIS_SIGN[i]
        origins[i] = p
        R = R @ _elementary(_AXIS_ID[i], _AXIS_SIGN[i] * q[i])
    pe = p + R @ EE_OFFSET
    J = np.empty((6, 6))
    J[:3] = np.cross(axes, pe - origins).T
    J[3:] = axes.T
    return R, pe, J


def jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (6, 6) of ee_site, rows [v; w] in the base frame."""
    return fk_jacobian(np.asarray(q, dtype=np.float64))[2]


def ik(R_t: np.ndarray, p_t: np.ndarray, seed: np.ndarray, iters: int = 30,
       damping: float = 1e-3, tol_pos: float = 1e-4, tol_rot: float = 1e-3):
    """Damped least squares from ``seed`` (6,). Returns (q, pos_err_m, rot_err_rad).

    Clamps to joint limits each step. The caller decides what residual is acceptable --
    on the robot an unreachable or ambiguous target must hold, not guess.
    """
    q = np.array(seed, dtype=np.float64)
    eye = damping * np.eye(6)
    for _ in range(iters):
        R, p, J = fk_jacobian(q)
        e_p = p_t - p
        e_r = R @ rotvec_from_matrix(R.T @ R_t)
        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
            break
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + eye, e)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0] - IK_LIMIT_MARGIN, JOINT_LIMITS[:, 1] + IK_LIMIT_MARGIN)
    R, p, _ = fk_jacobian(q)
    return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rotvec_from_matrix(R.T @ R_t)))


def chunk_to_joints(q0: np.ndarray, rel: np.ndarray, **ik_kw):
    """q0 (6,), rel (K, 6) -> (q (K, 6), pos_err (K,), rot_err (K,)). Each step seeds the next."""
    R, p = absolute_pose(q0, rel)
    out = np.zeros((len(rel), 6))
    ep = np.zeros(len(rel))
    er = np.zeros(len(rel))
    seed = np.asarray(q0, dtype=np.float64)
    for k in range(len(rel)):
        seed, ep[k], er[k] = ik(R[k], p[k], seed, **ik_kw)
        out[k] = seed
    return out, ep, er
