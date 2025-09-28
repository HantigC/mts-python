from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np


class _homo_dim_check(ABC):

    def __call__(self, xs: Union[np.ndarray, List], axis: int = 0) -> np.ndarray:

        if isinstance(xs, list):
            xs = np.array(xs)

        if xs.ndim == 2:
            if axis == 0:
                return self.on_two_zero(xs)
            elif axis == 1:
                return self.on_two_one(xs)
            else:
                raise ValueError(
                    f"When `xs` is two-dimensional, `axis` should 0 or 1, not {axis}"
                )
        elif xs.ndim == 1:
            if axis != 0:
                raise ValueError(
                    f"When `xs` is one-dimensional, `axis` should 0, not {axis}"
                )
            return self.on_zero(xs)

        raise ValueError(f"")

    @abstractmethod
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class _drop_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs[:, :-1]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1]


class _scale_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[:, -1, None]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[:-1]


class _from_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs[:, :-1] / xs[:, -1, None]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1] / xs[-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1] / xs[-1]


class _to_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return np.r_["1", xs, np.ones((xs.shape[0], 1))]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return np.r_["0", xs, np.ones((1, xs.shape[1]))]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return np.r_[xs, 1]


drop_homogenous = _drop_homogenous()
scale_homogenous = _scale_homogenous()
from_homogenous = _from_homogenous()
to_homogenous = _to_homogenous()
