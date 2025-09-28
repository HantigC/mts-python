from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from tqdm.auto import tqdm

from mts.geometry.transform import view_from_Rt
from mts.geometry.triangulation import linear
from mts.optim.jr.rigid import jr
from mts.types import NPMatrix3x3f, NPMatrix4x4f, NPVector3f
from mts.util.np import add_col


@dataclass
class Rigid3D:
    R: NPMatrix3x3f
    t: NPVector3f

    @property
    def inv_t(self) -> NPVector3f:
        return -self.R.T @ self.t

    @property
    def inv_R(self) -> NPMatrix3x3f:
        return self.R.T

    @property
    def xaxis(self) -> NPVector3f:
        return self.R[0]

    @property
    def yaxis(self) -> NPVector3f:
        return self.R[1]

    @property
    def zaxis(self) -> NPVector3f:
        return self.R[2]

    @property
    def Rt4x4(self) -> NPMatrix4x4f:
        return view_from_Rt(self.R, self.t)

    def z_of(self, point: NPVector3f) -> float:
        return np.dot(self.R[2], point) + self.t[2]

    @classmethod
    def from_identity(cls) -> Rigid3D:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_inv_rigid(cls, rigid_3d: Rigid3D) -> Rigid3D:
        inv_R = rigid_3d.R.copy().T
        inv_t = -inv_R @ rigid_3d.t
        return cls(inv_R, inv_t)

    def __mul__(self, other: Rigid3D) -> Rigid3D:
        if isinstance(other, Rigid3D):
            mul_R = self.R @ other.R
            mul_t = self.R @ other.t + self.t
            return Rigid3D(mul_R, mul_t)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1 and len(other) == 3:
                return self.R @ other + self.t
            elif other.ndim == 2 and other.shape[1] == 3:
                return other @ self.R.T + self.t
        elif isinstance(other, (tuple, list)) and len(other) == 3:
            return self.R @ other + self.t

    @classmethod
    def from_T(cls, T: NPMatrix4x4f) -> Rigid3D:
        R, t = T[:3, :3], T[:3, 3]
        return cls(R, t)

    @classmethod
    def Exp(cls, tr: np.ndarray):
        T = exp_se3(tr)
        return cls.from_T(T)


def skew(w):
    """Return the skew-symmetric matrix of a 3-vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w):
    """Exponential map for so(3) -> SO(3)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    w_hat = skew(w)
    R = (
        np.eye(3)
        + (np.sin(theta) / theta) * w_hat
        + ((1 - np.cos(theta)) / (theta**2)) * (w_hat @ w_hat)
    )
    return R


def left_jacobian_SO3(w):
    """Left Jacobian of SO(3)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    w_hat = skew(w)
    J = (
        np.eye(3)
        + (1 - np.cos(theta)) / (theta**2) * w_hat
        + (theta - np.sin(theta)) / (theta**3) * (w_hat @ w_hat)
    )
    return J


def exp_se3(xi):
    """
    Exponential map for se(3) -> SE(3).
    xi: 6-vector (v, w), where
    - w (3,) = angular part
    - v (3,) = linear part
    Returns: (4,4) homogeneous matrix in SE(3).
    """
    v = xi[:3]
    w = xi[3:]
    R = exp_so3(w)
    J = left_jacobian_SO3(w)
    p = J @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def compute_depth_mask(
    points_3d,
    st_view,
    nd_view,
    min_depth=0,
    max_depth=np.inf,
) -> np.ndarray:
    zs_transform = np.stack([st_view[2], nd_view[2]])
    points_3d_h = add_col(points_3d, 1)
    both_view_depths = points_3d_h @ zs_transform.T
    mask = (both_view_depths >= min_depth).all(axis=1) & (
        both_view_depths <= max_depth
    ).all(axis=1)
    return mask


def check_cheirality_PP(st_view, nd_view, st_points, nd_points):
    points_3D = linear.two_cameras(st_points, nd_points, st_view, nd_view)
    depth_mask = compute_depth_mask(points_3D, st_view, nd_view)
    return points_3D, depth_mask


def check_cheirality_RRtt(R1, t1, R2, t2, st_points, nd_points):
    st_view = view_from_Rt(R1, t1)
    nd_view = view_from_Rt(R2, t2)
    return check_cheirality_PP(st_view, nd_view, st_points, nd_points)


def check_cheirality_Rt(R, t, st_points, nd_points):
    st_view = np.eye(4)
    nd_view = view_from_Rt(R, t)
    return check_cheirality_PP(st_view, nd_view, st_points, nd_points)


def compute_pose(E, st_points, nd_points):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.squeeze()
    pose_comb = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    best_point_3D = None
    max_visible = -1

    best_R = None
    best_t = None
    best_mask = None

    for rotation, translation in pose_comb:
        point_3D, mask = check_cheirality_Rt(
            rotation,
            translation,
            st_points,
            nd_points,
        )
        num_visible = mask.sum()
        if num_visible > max_visible:
            best_point_3D = point_3D
            max_visible = num_visible
            best_R = rotation
            best_t = translation
            best_mask = mask

    return best_R, best_t, best_point_3D, best_mask


def gd_pnp(
    K,
    pose: Rigid3D,
    image_points: np.ndarray,
    world_points: np.ndarray,
    iterations: int = 100,
    learning_rate: float = 0.000001,
    cost_th: float = 0.02,
):
    last_cost = np.inf
    with tqdm(total=iterations) as tbar:

        for _ in range(iterations):
            js, es = jr(K, pose, image_points, world_points)
            dxs = -(js.transpose(2, 1, 0) @ es[:, :, np.newaxis]).squeeze()
            dx = dxs.mean(0)
            dx = learning_rate * dx

            cost = np.linalg.norm(es, axis=1).mean()
            if cost - cost_th > last_cost:
                break
            last_cost = cost
            tbar.set_postfix({"cost": cost})
            tbar.update()

            pose = Rigid3D.Exp(dx) * pose
    return pose


def gn_pnp(
    K,
    pose: Rigid3D,
    image_points: np.ndarray,
    world_points: np.ndarray,
    iterations: int = 100,
    cost_th: float = 0.02,
):
    last_cost = np.inf
    with tqdm(total=iterations) as tbar:
        for _ in range(iterations):
            js, es = jr(K, pose, image_points, world_points)
            Hs = js.transpose(2, 1, 0) @ js.transpose(2, 0, 1)
            dxs = -(js.transpose(2, 1, 0) @ es[:, :, np.newaxis]).squeeze()
            H = Hs.sum(0)
            dx = dxs.sum(0)
            dx = np.linalg.solve(H, dx)
            cost = np.linalg.norm(es, axis=1).mean()
            if cost - cost_th > last_cost:
                break
            last_cost = cost

            tbar.set_postfix({"cost": cost})
            tbar.update()

            pose = Rigid3D.Exp(dx) * pose
    return pose

