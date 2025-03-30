import numpy as np
from mts.types import L, Size, NPVector3f
from typing import Tuple


def angle_between(
    camera_center1: NPVector3f,
    camera_center2: NPVector3f,
    points_3d: np.ndarray[Tuple[Size, L[3]], np.float32],
    as_degree: bool = False,
) -> np.ndarray[Size, np.float32]:
    if points_3d.ndim == 1:
        points_3d = points_3d[np.newaxis]
    base = np.sqrt(np.sum((camera_center1 - camera_center2) ** 2))
    ray1 = np.sqrt(np.sum((points_3d - camera_center1) ** 2, axis=1))
    ray2 = np.sqrt(np.sum((points_3d - camera_center2) ** 2, axis=1))
    denominator = 2 * ray1 * ray2
    numerator = ray1**2 + ray2**2 - base**2
    angles = np.where(
        denominator != 0,
        np.abs(np.acos(numerator / denominator)),
        0,
    )

    angles = np.where(
        angles < np.pi - angles,
        angles,
        np.pi - angles,
    )
    if as_degree:
        angles = np.rad2deg(angles)
    return angles
