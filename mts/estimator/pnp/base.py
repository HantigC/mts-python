from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from mts.types import NPManyVector2f, NPManyVector3f, NPMatrix3x3f, NPVector3f


@dataclass
class PnPSummary:
    R: NPMatrix3x3f
    t: NPVector3f


PnPType = TypeVar("T", bound=PnPSummary)


class BasePnPEstimator(ABC, Generic[PnPType]):

    @abstractmethod
    def estimate_normalized(
        self,
        world_points: NPManyVector3f,
        camera_points: NPManyVector3f,
    ) -> PnPType:
        pass

    @abstractmethod
    def estimate(
        self,
        world_points: NPManyVector3f,
        camera_points: NPManyVector2f,
        K: NPMatrix3x3f,
    ) -> PnPType:
        pass
