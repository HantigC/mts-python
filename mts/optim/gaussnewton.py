from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from tqdm.auto import tqdm

TermType = TypeVar("TermType")
ParamType = TypeVar("ParamType")


class BaseJR(ABC, Generic[TermType, ParamType]):

    @abstractmethod
    def compute_jr(
        self,
        term: TermType,
        param: ParamType,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    @abstractmethod
    def to_second_order(
        self,
        jacobians: np.ndarray,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def update_term(
        self,
        term: TermType,
        gradient: np.ndarray,
        param: ParamType,
    ) -> TermType:
        pass


class ToSecondOrderMixin:

    def to_second_order(
        self,
        jacobians: np.ndarray,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        jacobians_T = jacobians.transpose(0, 2, 1)
        hessians = jacobians_T @ jacobians
        gradients = -(jacobians_T @ residuals[:, :, np.newaxis]).squeeze()
        hessian = hessians.sum(0)
        gradient = gradients.sum(0)
        return hessian, gradient


JRTType = TypeVar("JRTType", bound=BaseJR)


class GaussNewton(Generic[TermType, ParamType]):
    def __init__(self, jr: BaseJR[TermType, ParamType]):
        self.jr = jr

    def compute(
        self,
        term: TermType,
        param: ParamType,
        iterations: int,
        cost_threshold: float,
        verbose: bool = True,
    ) -> TermType:
        last_cost = np.inf
        with tqdm(total=iterations, disable=not verbose) as tbar:
            for _ in range(iterations):
                js, es, cost = self.jr.compute_jr(term, param)
                hessian, gradient = self.jr.to_second_order(js, es)
                update_step = np.linalg.solve(hessian, gradient)
                if cost - cost_threshold > last_cost:
                    break
                last_cost = cost

                tbar.set_postfix({"cost": cost})
                tbar.update()
                term = self.jr.update_term(term, update_step, param)
        return term
