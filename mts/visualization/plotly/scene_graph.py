from dataclasses import dataclass
from typing import List
from hloc.visualization import plot_images, plot_keypoints
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from mts.model.image import Image, ImageId, ImageType
from mts.pose.rigid import Rigid3D
from mts.sfm.reconstruction import Reconstruction
from mts.sfm.scene_graph import ReconstructedPoint, SceneGraph
from mts.visualization.plotly.camera import render_axes
from mts.visualization.plotly.common import create_new_figure, init_figure
from mts.visualization.plotly.rigid import render_rigi3d_pose

from plotly import graph_objects as go


@dataclass
class SceneGraphViz:
    scene_graph: SceneGraph

    def plot_tracks(self, track_id: int) -> None:
        images = []
        kps_idxs = []
        for image_id, kp_num in self.scene_graph.points[track_id].track:
            image = self.scene_graph.images[image_id]
            images.append(image.img)
            kps_idxs.append(image.keypoints[None, kp_num])
        plot_images(images)
        plot_keypoints(kps_idxs, ps=40)

    def plot_graph(self):
        graph = nx.Graph()

        for image_id in self.scene_graph.image_ids:
            graph.add_node(image_id)

        matches_map = {}
        for pair in self.scene_graph.pairs.values():
            st_image_id, nd_image_id = pair.image_pair_ids
            graph.add_edge(st_image_id, nd_image_id)
            matches_map[st_image_id, nd_image_id] = len(pair.matches)

        plt.figure()
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos=pos,
            with_labels=True,
            edge_color=list(matches_map.values()),
            font_weight="bold",
        )
        nx.draw_networkx_edge_labels(
            graph,
            pos=pos,
            edge_labels=matches_map,
            font_size=6,
        )

    @create_new_figure
    def plot_points_at_camera(
        self,
        image: ImageType,
        fig=None,
    ) -> go.Figure:
        if isinstance(image, ImageId):
            image = self.scene_graph.images[image]

        xs, ys, zs = [], [], []
        points_colors = []
        for _, inv_track in self.scene_graph.keypoint_invtracks[image.image_id].items():
            reconstructed_point = self.scene_graph.points[inv_track.idx]
            if reconstructed_point.loc is not None:
                point = image.to_camera(reconstructed_point.loc.as_np())
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
                points_colors.append(reconstructed_point.color.as_tuple())

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    # symbol="x",
                    size=1,
                    color=points_colors,
                ),
            )
        )
        render_rigi3d_pose(Rigid3D.from_identity(), fig=fig)

        return fig

    @create_new_figure
    def plot_visible_points(
        self,
        image: ImageType,
        fig=None,
        in_camera: bool = False,
    ) -> go.Figure:
        if isinstance(image, ImageId):
            image = self.scene_graph.images[image]

        xs, ys, zs = [], [], []
        points_colors = []
        for _, inv_track in self.scene_graph.keypoint_invtracks[image.image_id].items():
            reconstructed_point = self.scene_graph.points[inv_track.idx]
            if reconstructed_point.loc is not None:
                point = reconstructed_point.loc.as_tuple()
                if in_camera:
                    point = image.pose * point

                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
                points_colors.append(reconstructed_point.color.as_tuple())

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    # symbol="x",
                    size=1,
                    color=points_colors,
                ),
            )
        )
        return fig

    def plot_visible_points_for_images(self, images: List[ImageId]):
        pass

    @create_new_figure
    def plot_visible_kps(self, image_id: ImageId | Image, fig=None) -> go.Figure:

        if isinstance(image_id, Image):
            image_id = image_id.image_id
        image = self.scene_graph.images[image_id]
        if image.pose is None:
            raise ValueError(f"Image with id={image_id} has no pose")

        xs, ys, zs = [], [], []
        points_colors = []
        kp_xs, kp_ys, kp_zs = [], [], []
        kp_colors = []
        world_kp = image.keypoints_w
        for kp_idx, inv_track in self.scene_graph.keypoint_invtracks[image_id].items():
            reconstructed_point = self.scene_graph.points[inv_track.idx]
            if reconstructed_point.loc is not None:
                xs.append(reconstructed_point.loc.x)
                ys.append(reconstructed_point.loc.y)
                zs.append(reconstructed_point.loc.z)
                points_colors.append(reconstructed_point.color.as_tuple())

                x, y, z = world_kp[kp_idx]
                kp_xs.append(x)
                kp_ys.append(y)
                kp_zs.append(z)

                kp_colors.append(image.kp_color(kp_idx).as_tuple())

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    # symbol="x",
                    size=1,
                    color=points_colors,
                ),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=kp_xs,
                y=kp_ys,
                z=kp_zs,
                mode="markers",
                marker=dict(
                    # symbol="x",
                    size=1,
                    color=kp_colors,
                ),
            )
        )
        render_rigi3d_pose(image.pose, fig=fig)

        return fig


@create_new_figure
def render_reconstructed_points(
    reconstructed_points: List[ReconstructedPoint],
    strict: bool = True,
    *,
    fig: go.Figure = None,
    **kwargs,
) -> go.Figure:
    xs, ys, zs = [], [], []
    points_colors = []
    for reconstructed_point in reconstructed_points:
        reconstructed_point.loc
        if reconstructed_point.loc is None:
            if strict:
                raise ValueError("Every reconstructed point should have a location")

        xs.append(reconstructed_point.loc.x)
        ys.append(reconstructed_point.loc.y)
        zs.append(reconstructed_point.loc.z)
        points_colors.append(reconstructed_point.color.as_tuple())

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                # symbol="x",
                size=1,
                color=points_colors,
            ),
            **kwargs,
        )
    )
    return fig


@create_new_figure
def render_reconstruction(
    reconstruction: Reconstruction,
    *,
    fig: go.Figure = None,
) -> go.Figure:
    reconstruction.points
    # render_two_view(fig, starting_pair)
    xs, ys, zs = [], [], []
    points_colors = []
    for tracklet in reconstruction.points.values():
        xs.append(tracklet.loc.x)
        ys.append(tracklet.loc.y)
        zs.append(tracklet.loc.z)
        points_colors.append(tracklet.color.as_tuple())

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                # symbol="x",
                size=1,
                color=points_colors,
            ),
        )
    )
    for new_image in reconstruction.images:
        new_pos = -np.linalg.inv(new_image.pose.R) @ new_image.pose.t
        xaxis, yaxis, zaxis = new_image.pose.R

        render_axes(
            fig,
            new_pos,
            xaxis,
            yaxis,
            zaxis,
            name=f"image {new_image.image_id}",
        )

    return fig
