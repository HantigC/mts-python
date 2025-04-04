from dataclasses import dataclass
from typing import Tuple, TypeAlias

import numpy as np

from mts.geometry.transform import to_camera_coord, to_world_coord_Rt
from mts.model.rgb import RGB
from mts.pose.rigid import Rigid3D
from mts.types import (
    L,
    NPMatrix3x3f,
    NPVector2f,
    NPVector3f,
    Size,
    NPManyVector3f,
    NPManyVector2f,
)
from mts.util.np import add_col


@dataclass
class Image:
    img: np.ndarray
    image_id: int = -1
    K: NPMatrix3x3f = None
    keypoints: np.ndarray[Tuple[Size, L[2]], np.float32] = None
    descriptors: np.ndarray[Tuple[Size, Size], np.float32] = None
    pose: Rigid3D = None

    @property
    def camera_keypoints(self):
        return to_camera_coord(self.K, self.keypoints_h)

    @property
    def keypoints_h(self):
        return add_col(self.keypoints, 1)

    @property
    def keypoints_w(self):
        if self.pose is None:
            raise AttributeError("Camera doesn't have a pose...yet:)")
        return to_world_coord_Rt(self.K, self.pose.R, self.pose.t, self.keypoints_h)

    def kp_color(self, kp_idx: int) -> RGB:
        x, y = self.keypoints[kp_idx]
        r, g, b = self.img[int(y), int(x)]
        return RGB(r, g, b)

    def to_camera(self, world_point: NPVector3f | NPManyVector3f) -> NPVector3f:
        if world_point.ndim == 1:
            world_point = world_point[np.newaxis]
        elif world_point.ndim != 2:
            raise ValueError(f"The passed value should be of type: {NPManyVector3f}")

        camera_points = world_point @ self.pose.R.T   + self.pose.t
        camera_points = np.squeeze(camera_points)
        return camera_points

    def to_camera(self, world_point: NPVector3f) -> NPVector3f:
        camera_point = self.pose.R @ world_point + self.pose.t
        return camera_point

    def project_points(self, points_3d: NPManyVector3f) -> NPManyVector2f:
        projected_2d_h = (points_3d @ self.pose.R.T + self.pose.t) @ self.K.T
        projected_2d = projected_2d_h[:, :2] / projected_2d_h[:, 2:]
        return projected_2d

    def project_point(self, x_3d: NPVector3f) -> NPVector2f:
        x_2d_h = self.K @ (self.pose.R @ x_3d + self.pose.t)
        x_2d = x_2d_h[:2] / x_2d_h[2]
        return x_2d


ImageId: TypeAlias = int
ImageType = Image | ImageId
