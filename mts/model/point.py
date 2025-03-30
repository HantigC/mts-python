from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple

import numpy as np

from mts.types import NPManyVector3f, NPVector3f, Number


@dataclass
class Point3D(Generic[Number]):
    x: Number
    y: Number
    z: Number

    @classmethod
    def from_numpy(cls, point_3d_np: NPVector3f) -> Point3D:
        obj = cls(
            point_3d_np[0],
            point_3d_np[1],
            point_3d_np[2],
        )
        return obj

    def as_np(self) -> NPVector3f:
        return np.array([self.x, self.y, self.z])

    def as_tuple(self) -> Tuple[Number, Number, Number]:
        return (self.x, self.y, self.z)

    def as_list(self) -> List[Number]:
        return [self.x, self.y, self.z]

    def as_dict(self) -> Dict[str, Number]:
        return dict(x=self.x, y=self.y, z=self.z)


def points_3f_from_np(points_3d: NPManyVector3f) -> List[Point3D]:
    return [Point3D(x, y, z) for x, y, z in points_3d]
