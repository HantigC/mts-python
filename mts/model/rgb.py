from __future__ import annotations
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from mts.types import L


@dataclass
class RGB:
    r: float
    g: float
    b: float

    @classmethod
    def from_numpy(cls, point_3d_np: np.ndarray[L[3], np.floating]) -> RGB:
        obj = cls(
            point_3d_np[0],
            point_3d_np[1],
            point_3d_np[2],
        )
        return obj

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.r, self.g, self.b
