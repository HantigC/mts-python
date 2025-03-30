import numpy as np
from mts.util.np import add_row
from mts.types import (
    NPMatrix4x4f,
    NPVector3f,
    NPMatrix3x3f,
    NPManyVector3f,
)


def to_camera_coord(
    K: NPMatrix3x3f,
    points: NPManyVector3f,
) -> NPManyVector3f:
    K_inv = np.linalg.inv(K)
    camera_points = points @ K_inv.T
    return camera_points


def view_from_Rt(
    R: NPMatrix3x3f,
    t: NPVector3f,
) -> NPMatrix4x4f:
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def to_world_coord_Rt(
    K: NPMatrix3x3f,
    R: NPMatrix3x3f,
    t: NPVector3f,
    points: NPManyVector3f,
) -> NPManyVector3f:
    inv_R = R.T
    inv_t = -inv_R @ t
    inv_view = view_from_Rt(inv_R, inv_t)
    world_coords = inv_view @ add_row(np.linalg.inv(K) @ points.T, 1)
    world_coords = world_coords[:3].T
    return world_coords
