from typing import Union, List
import numpy as np
from geom.vec import from_homogenous


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
