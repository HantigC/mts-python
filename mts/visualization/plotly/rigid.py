from plotly import graph_objects as go

from mts.pose.rigid import Rigid3D
from mts.visualization.plotly.camera import render_axes
from mts.visualization.plotly.common import create_new_figure


@create_new_figure
def render_Rt(
    R,
    t,
    fig: go.Figure = None,
    **kwargs,
) -> go.Figure:
    xaxis, yaxis, zaxis = R
    return render_axes(fig, t, xaxis, yaxis, zaxis, **kwargs)


@create_new_figure
def render_rigi3d_pose(
    pose: Rigid3D,
    fig: go.Figure = None,
    **kwargs,
) -> go.Figure:
    xaxis, yaxis, zaxis = pose.R
    return render_axes(fig, pose.inv_t, xaxis, yaxis, zaxis, **kwargs)
