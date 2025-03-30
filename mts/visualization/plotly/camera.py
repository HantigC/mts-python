from plotly import graph_objects as go

from util.dict import extract_kwargs


def render_axes(
    fig,
    position,
    xaxis,
    yaxis,
    zaxis,
    scale=1,
    **kwargs,
):
    named_kwargs = extract_kwargs(kwargs, ["xaxis", "yaxis", "zaxis"], merge=True)
    xaxis_kwargs = named_kwargs["xaxis"]
    yaxis_kwargs = named_kwargs["yaxis"]
    zaxis_kwargs = named_kwargs["zaxis"]
    xaxis_kwargs.setdefault("color", "rgb(255, 0, 0)")
    yaxis_kwargs.setdefault("color", "rgb(0, 255, 0)")
    zaxis_kwargs.setdefault("color", "rgb(0, 0, 255)")

    xaxis_kwargs.setdefault("name", "xaxis")
    yaxis_kwargs.setdefault("name", "yaxis")
    zaxis_kwargs.setdefault("name", "zaxis")

    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * xaxis[0]],
            y=[position[1], position[1] + scale * xaxis[1]],
            z=[position[2], position[2] + scale * xaxis[2]],
            mode="lines",
            name=xaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=xaxis_kwargs["color"]),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * yaxis[0]],
            y=[position[1], position[1] + scale * yaxis[1]],
            z=[position[2], position[2] + scale * yaxis[2]],
            mode="lines",
            name=yaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=yaxis_kwargs["color"]),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * zaxis[0]],
            y=[position[1], position[1] + scale * zaxis[1]],
            z=[position[2], position[2] + scale * zaxis[2]],
            mode="lines",
            name=zaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=zaxis_kwargs["color"]),
        )
    )
    return fig
