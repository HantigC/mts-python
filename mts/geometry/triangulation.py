from typing import List, Union

import numpy as np

from mts.geometry.vec import from_homogenous


def to_2d(x: np.ndarray):
    if x.ndim == 1:
        return x[np.newaxis, :]
    elif x.ndim == 2:
        return x

    else:
        raise ValueError(f"`x` should have either 1 or 2 dims, not {x.ndim} dims")


def pairwise_dot(x: np.array, y: np.array, axis: int = 1) -> np.array:
    return (x * y).sum(axis=axis)


def mid_point(
    p: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    return_on_rays: bool = False,
) -> np.ndarray:
    p = to_2d(p)
    r = to_2d(r)
    q = to_2d(q)
    s = to_2d(s)

    ss = pairwise_dot(s, s, axis=1)
    rr = pairwise_dot(r, r, axis=1)
    rs = pairwise_dot(r, s, axis=1)

    qr = pairwise_dot(q, r, axis=1)
    pr = pairwise_dot(p, r, axis=1)
    ps = pairwise_dot(p, s, axis=1)
    qs = pairwise_dot(q, s, axis=1)

    miu = (ss * (qr - pr) + rs * (ps - qs)) / (rr * ss - rs**2)
    sigma = (rr * (ps - qs) + rs * (qr - pr)) / (rr * ss - rs**2)
    P = p + r * sigma[:, np.newaxis]
    Q = q + s * miu[:, np.newaxis]
    H = (P + Q) / 2
    if return_on_rays:
        return H, P, Q
    return H


class _Linear:

    def two_cameras(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        cam1: np.ndarray,
        cam2: np.ndarray,
    ) -> np.ndarray:
        triangulated_points = self.multi_points([points1, points2], [cam1, cam2])
        return triangulated_points

    def multi_points(
        self,
        points: Union[np.ndarray, List[np.ndarray]],
        cameras: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[np.ndarray, List[np.ndarray]]:

        points = np.stack(points, axis=1)
        cams = np.stack(cameras)

        sistems = points[..., :2, np.newaxis] * cams[:, 2, np.newaxis] - cams[:, :2]
        sistems = sistems.reshape(-1, sistems.shape[-2] * sistems.shape[-3], 4)

        triangulated_points = np.linalg.svd(sistems)
        triangulated_points = triangulated_points.Vh[:, 3, :]
        triangulated_points = from_homogenous(triangulated_points, axis=1)
        return triangulated_points

    def multi_cameras(
        self,
        points: Union[np.ndarray, List[np.ndarray]],
        cameras: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[np.ndarray, List[np.ndarray]]:

        points = np.stack(points, axis=0)
        cameras = np.stack(cameras)
        system = points[:, :2, np.newaxis] * cameras[:, np.newaxis, 2] - cameras[:, :2]

        system = system.reshape(-1, 4)

        USVt = np.linalg.svd(system)
        triangulated_point = USVt.Vh[3, :]
        triangulated_point = from_homogenous(triangulated_point)
        return triangulated_point

    __call__ = two_cameras


linear = _Linear()
