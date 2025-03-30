from abc import ABC, abstractmethod
import numpy as np
from mts.types import Columns, L
from typing import Any, Tuple



class BaseMatcher(ABC):

    @abstractmethod
    def match(
        self,
        st_descriptors: np.ndarray[Tuple[Any, Columns], np.float32],
        nd_descriptors: np.ndarray[Tuple[Any, Columns], np.float32],
    ) -> np.ndarray[Tuple[Any, L[2]], np.uint32]:
        pass
