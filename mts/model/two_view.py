from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, TypeAlias

import cv2
import numpy as np

from mts.geometry.transform import to_camera_coord, view_from_Rt
from mts.geometry.trigonometry import angle_between
from mts.model.image import Image, ImageId
from mts.pose.rigid import Rigid3D, compute_pose, compute_depth_mask
from mts.util.np import add_col

PairId: TypeAlias = int


@dataclass
class TwoViewPair:
    st_image: Image
    nd_image: Image

    matches: np.ndarray = None
    relative_pose: Rigid3D = None
    E: np.ndarray = None
    F: np.ndarray = None
    H: np.ndarray = None

    points3D: np.ndarray = None

    @cached_property
    def pair_id(self):
        return image_ids_to_pair_id(self.st_image.image_id, self.nd_image.image_id)

    @cached_property
    def image_pair_ids(self) -> Tuple[ImageId, ImageId]:
        return (self.st_image.image_id, self.nd_image.image_id)

    @cached_property
    def st_keypoints(self):
        return self.st_image.keypoints[self.matches[:, 0]]

    @cached_property
    def nd_keypoints(self):
        return self.nd_image.keypoints[self.matches[:, 1]]

    @cached_property
    def st_colors(self):
        xy = self.st_image.keypoints[self.matches[:, 0]].T
        x, y = xy.astype(np.int32)
        pixels = self.st_image.img[y, x]
        return pixels

    @property
    def angle(self):
        angles = angle_between(
            [0, 0, 0],
            self.relative_pose.inv_t,
            self.points3D,
            as_degree=True,
        )
        return np.mean(angles)

    @cached_property
    def nd_colors(self):
        xy = self.nd_image.keypoints[self.matches[:, 1]].T
        x, y = xy.astype(np.int32)
        pixels = self.nd_image.img[y, x]
        return pixels

    @cached_property
    def st_keypoints_camera(self):
        return to_camera_coord(self.st_image.K, self.st_keypoints_h)

    @cached_property
    def nd_keypoints_camera(self):
        return to_camera_coord(self.nd_image.K, self.nd_keypoints_h)

    @cached_property
    def st_keypoints_h(self):
        return add_col(self.st_keypoints, 1)

    @cached_property
    def nd_keypoints_h(self):
        return add_col(self.nd_keypoints, 1)


MAX_IMAGE_ID = 2**31 - 1


def compute_two_view(match, st_image: Image, nd_image: Image) -> TwoViewPair:
    matches = match(st_image.descriptors, nd_image.descriptors)
    F, mask = cv2.findFundamentalMat(
        st_image.keypoints[matches[:, 0]],
        nd_image.keypoints[matches[:, 1]],
        cv2.FM_RANSAC,
    )
    if mask is None:
        return None
    matches = matches[mask.ravel().astype(np.bool)]
    E = st_image.K.T @ F @ nd_image.K
    R, t, points3D, mask = compute_pose(
        E,
        st_image.camera_keypoints[matches[:, 0], :2],
        nd_image.camera_keypoints[matches[:, 1], :2],
    )

    points3D = points3D[mask]
    matches = matches[mask]
    mask = compute_depth_mask(
        points3D,
        np.eye(4),
        view_from_Rt(R, t),
        0.1,
        20.0,
    )

    points3D = points3D[mask]
    matches = matches[mask]

    two_view_pair = TwoViewPair(
        st_image=st_image,
        nd_image=nd_image,
        matches=matches,
        F=F,
        E=E,
        relative_pose=Rigid3D(R, t),
        points3D=points3D,
    )
    return two_view_pair


def image_ids_to_pair_id(image_id1: ImageId, image_id2: ImageId) -> PairId:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id: PairId) -> Tuple[ImageId, ImageId]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2
