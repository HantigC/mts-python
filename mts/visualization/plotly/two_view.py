
def render_two_view(fig: go.Figure, two_view: TwoViewPair):
    render_axes(fig, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    render_axes(
        fig,
        two_view.relative_pose.inv_t,
        two_view.relative_pose.xaxis,
        two_view.relative_pose.yaxis,
        two_view.relative_pose.zaxis,
    )
    xs, ys, zs = two_view.points3D.T
    image_x, image_y = two_view.st_keypoints.astype(np.uint32).T
    rgbs = two_view.st_image.img[image_y, image_x]
    

    scatter_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(symbol='x', size=1, color=rgbs),
    )
    fig.add_trace(scatter_trace)
    return fig