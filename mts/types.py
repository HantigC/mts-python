from __future__ import annotations

from typing import Literal, TypeAlias, TypeVar

import numpy as np

L = Literal

Number = TypeVar("Number", float, int)
Size = TypeVar("Size", bound=int)

Rows = TypeVar("Rows", bound=int)
Columns = TypeVar("Columns", bound=int)
DType = TypeVar("DType")
Id: TypeAlias = int


NPMatrix3x3f = np.ndarray[(3, 3), np.float32]
NPMatrix4x4f = np.ndarray[(4, 4), np.float32]

NPVector3f = np.ndarray[3, np.float32]
NPVector4f = np.ndarray[4, np.float32]
NPVector2f = np.ndarray[2, np.float32]


NPManyVector3f = np.ndarray[(-1, 3), np.float32]
NPManyVector2f = np.ndarray[(-1, 2), np.float32]
NPManyVector4f = np.ndarray[(-1, 4), np.float32]
