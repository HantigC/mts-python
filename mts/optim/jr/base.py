from typing import NamedTuple
import numpy as np


class JacobianResiduals(NamedTuple):
    jacobians: np.ndarray
    residuals: np.ndarray