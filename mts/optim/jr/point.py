from typing import NamedTuple

import numpy as np

from mts.optim.gaussnewton import BaseJR, ToSecondOrderMixin
from mts.optim.jr.rigid import PoseParam
from mts.pose.rigid import Rigid3D
from mts.types import NPManyVector2f, NPManyVector3f, NPMatrix3x3f, NPVector2f, NPVector3f


class PointParam(NamedTuple):
    Ks: list[NPMatrix3x3f]
    image_points: list[NPManyVector2f]
    poses: list[Rigid3D]


class PointJR(ToSecondOrderMixin, BaseJR[NPVector3f, PointParam]):

    def compute_jr(
        self,
        point: NPVector3f,
        param: PointParam,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        jacobians, residuals = jr(
            point,
            param.poses,
            param.image_points,
            param.Ks,
        )
        cost = np.linalg.norm(residuals, axis=1).mean()
        return jacobians, residuals, cost

    def update_term(
        self,
        point: NPManyVector3f,
        gradient: np.ndarray,
        param: PoseParam,
    ) -> NPManyVector3f:
        point = point + gradient
        return point


def jr(
    world_point: NPVector3f,
    poses: list[Rigid3D],
    image_points: list[NPVector2f],
    Ks: list[NPMatrix3x3f],
) -> tuple[np.ndarray, np.ndarray]:

    camera_points = []
    focals = []
    centers = []
    rotations = []
    for pose, K in zip(poses, Ks):
        camera_point = pose * world_point
        camera_points.append(camera_point)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        focals.append([fx, fy])
        centers.append([cx, cy])
        rotations.append(pose.R)

    camera_points = np.array(camera_points)
    centers = np.array(centers)
    focals = np.array(focals)
    rotations = np.stack(rotations)
    image_points = np.stack(image_points)

    xc, yc, zc = camera_points.T

    fx, fy = focals.T
    cx, cy = centers.T

    inv_zc = 1 / zc
    inv_zc_squared = 1 / (zc * zc)

    projected_points = np.stack(
        [
            fx * xc * inv_zc + cx,
            fy * yc * inv_zc + cy,
        ],
        axis=1,
    )

    m = np.array(
        [
            [fx * inv_zc, np.zeros_like(fx), -fx * xc * inv_zc_squared],
            [np.zeros_like(fx), fy * inv_zc, -fy * yc * inv_zc_squared],
        ]
    )
    m = m.transpose(2, 0, 1)
    residuals = image_points - projected_points
    jacobians = -m @ rotations
    return jacobians, residuals
