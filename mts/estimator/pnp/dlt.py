from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from mts.estimator.pnp.base import BasePnPEstimator, PnPSummary
from mts.types import NPManyVector2f, NPManyVector3f, NPMatrix3x3f

LOGGER = logging.getLogger(__name__)

rot180_z = R.from_euler("z", 180, degrees=True).as_matrix()


def make_A(X, x, K, isNormalized=False):
    if X.shape[1] == 3:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    if x.shape[1] == 2:
        x = np.hstack((x, np.ones((x.shape[0], 1))))

    if isNormalized == False:
        x = np.linalg.inv(K).dot(x.T).T

    A = np.zeros((X.shape[0] * 2, 12))

    for i in range(X.shape[0]):
        A[i * 2, :] = np.concatenate((-X[i, :], np.zeros((4,)), x[i, 0] * X[i, :]))
        A[i * 2 + 1, :] = np.concatenate((np.zeros((4,)), -X[i, :], x[i, 1] * X[i, :]))
    return A


def linear_pnp_image(X, x, K, isNormalized=False):
    A = make_A(X, x, K, isNormalized)

    _, _, v = np.linalg.svd(A)
    P = v[-1, :].reshape((3, 4), order="C")
    H, h = P[:, :3], P[:, -1]
    world_t = -np.linalg.inv(H) @ h

    q, r = np.linalg.qr(np.linalg.inv(H))
    R = rot180_z @ q.T
    newK = rot180_z @ r / r[2, 2]
    newt = -R @ world_t

    return R, newt


def linear_pnp_camera(X, x, K, isNormalized=False):
    A = make_A(X, x, K, isNormalized)

    _, _, v = np.linalg.svd(A)
    P = v[-1, :].reshape((3, 4), order="C")
    R, t = P[:, :3], P[:, -1]

    u, s, v = np.linalg.svd(R)
    R = u.dot(v)
    t = t / s[0]

    if np.linalg.det(R) < 0:
        R = R * -1
        t = t * -1

    return R, t


def compute_projections(X, R, t, K):
    proj_h = K.dot(R.dot(X.T) + t)
    proj = cv2.convertPointsFromHomogeneous(proj_h.T)[:, 0, :]
    return proj


def compute_projection_error(x2d, x2dreproj):
    return np.mean(np.sqrt(np.sum((x2d - x2dreproj) ** 2, axis=-1)))


@dataclass
class RansacPnPSummary(PnPSummary):
    mask: np.ndarray
    projection_errors: np.ndarray


@dataclass
class RansacConfig:
    iters: int = field(default=3_000)
    no_points: int = field(default=6)
    outlier_thres: float = field(default=3)


class LinearPNPRansac(BasePnPEstimator[RansacPnPSummary]):

    def __init__(
        self,
        outlier_thres,
        iters,
        no_points=6,
        selection_criteria=None,
        **kwargs,
    ):
        self.outlier_thres = outlier_thres
        self.iters = iters
        self.no_points = no_points
        self.selection_criteria = selection_criteria

    def estimate_normalized(
        self,
        world_points: NPManyVector3f,
        camera_points: NPManyVector3f,
    ) -> RansacPnPSummary:
        raise NotImplementedError("`estimate_normalized` no yet implemented")

    def estimate(
        self,
        X: NPManyVector3f,
        x: NPManyVector2f,
        K: NPMatrix3x3f,
    ) -> RansacPnPSummary:

        bestR, bestt, bestmask, bestInlierCount = None, None, None, 0
        if len(X) < self.no_points:
            return None

        for i in range(self.iters):

            mask = np.random.randint(low=0, high=X.shape[0], size=(self.no_points,))
            Xiter = X[mask]
            xiter = x[mask]
            try:

                Riter, titer = linear_pnp_image(Xiter, xiter, K)
            except np.linalg.LinAlgError:
                continue
            # TODO: implement selection criteria with depth

            xreproj = compute_projections(X, Riter, titer[:, np.newaxis], K)
            errs = np.sqrt(np.sum((x - xreproj) ** 2, axis=-1))

            mask = errs < self.outlier_thres
            numInliers = np.sum(mask)

            if numInliers > bestInlierCount:
                bestInlierCount = numInliers
                bestR, bestt, bestmask = Riter, titer, mask

        if bestmask is None:
            return None
        X_best, x_best = X[bestmask], x[bestmask]
        try:
            R, t = linear_pnp_image(X_best, x_best, K)
        except Exception:
            LOGGER.exception("Linear PnP failed")
            return None
        else:
            xreproj = compute_projections(X_best, bestR, bestt[:, np.newaxis], K)
            errs = np.sqrt(np.sum((x_best - xreproj) ** 2, axis=-1))
            LOGGER.info("Best inliers: %d", bestInlierCount)
            return RansacPnPSummary(bestR, bestt, bestmask, errs)

    @classmethod
    def from_config(cls, config: RansacConfig) -> LinearPNPRansac:
        return cls(**config.__dict__)
