from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mts.geometry.transform import view_from_Rt
from mts.geometry.triangulation import linear
from mts.types import NPVector3f, NPMatrix4x4f, NPMatrix3x3f
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
            if other.dim == 1 and len(other) == 3:
                return self.R @ other + self.t
        elif isinstance(other, (tuple, list)) and len(other) == 3:
            return self.R @ other + self.t


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
