from typing import NamedTuple

import numpy as np

from ..gaussnewton import BaseJR, ToSecondOrderMixin

from mts.pose.rigid import Rigid3D
from mts.types import NPManyVector2f, NPManyVector3f, NPMatrix3x3f


class PoseParam(NamedTuple):
    K: NPMatrix3x3f
    image_points: NPManyVector2f
    world_points: NPManyVector3f


class PoseJR(ToSecondOrderMixin, BaseJR[Rigid3D, PoseParam]):

    def compute_jr(
        self,
        pose: Rigid3D,
        param: PoseParam,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        jacobians, residuals = jr(
            param.K,
            pose,
            param.image_points,
            param.world_points,
        )
        cost = np.linalg.norm(residuals, axis=1).mean()
        jacobians = jacobians.transpose(2, 0, 1)
        return jacobians, residuals, cost

    def update_term(
        self,
        pose: Rigid3D,
        gradient: np.ndarray,
        param: PoseParam,
    ) -> Rigid3D:
        pose = Rigid3D.Exp(gradient) * pose
        return pose


class _jr:

    def __call__(
        self,
        K: NPMatrix3x3f,
        pose: Rigid3D,
        image_points: np.ndarray,
        world_points: np.ndarray,
    ):
        camera_points = pose * world_points
        return self.in_camera(K, image_points, camera_points)

    def in_camera(self, K, image_points: np.ndarray, camera_points):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        xc, yc, zc = camera_points.T

        xc_squared = xc**2
        yc_squared = yc**2

        inv_zc = 1 / zc
        inv_zc_squared = 1 / (zc * zc)

        projected_points = np.stack(
            [
                fx * xc * inv_zc + cx,
                fy * yc * inv_zc + cy,
            ],
            axis=1,
        )

        js = np.array(
            [
                [
                    -fx * inv_zc,
                    np.zeros_like(inv_zc),
                    fx * xc * inv_zc_squared,
                    fx * xc * yc * inv_zc_squared,
                    -fx - fx * xc_squared * inv_zc_squared,
                    fx * yc * inv_zc,
                ],
                [
                    np.zeros_like(inv_zc),
                    -fy * inv_zc,
                    fy * yc * inv_zc_squared,
                    fy + fy * yc_squared * inv_zc_squared,
                    -fy * xc * yc * inv_zc_squared,
                    -fy * xc * inv_zc,
                ],
            ]
        )
        es = image_points - projected_points
        return js, es


jr = _jr()
