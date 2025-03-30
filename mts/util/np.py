import numpy as np


def add_col(array: np.array, x):
    xs = np.full((array.shape[0], 1), x)
    return np.r_["1", array, xs]


def add_row(array: np.array, x):
    xs = np.full((1, array.shape[1]), x)
    return np.r_["0", array, xs]
