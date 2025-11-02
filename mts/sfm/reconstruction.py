from dataclasses import dataclass, field
from typing import Dict, List
from collections import OrderedDict


from mts.model.image import Image
from mts.sfm.scene_graph import ReconstructedPoint, SceneGraph


@dataclass(init=False)
class Reconstruction:
    _scene_graph: SceneGraph = field()

    images: List[Image] = field(default_factory=list)
    points: OrderedDict[int, ReconstructedPoint] = field(default=None)

    def __init__(self, scene_graph: SceneGraph):
        self._scene_graph = scene_graph
        self.points = {}
        self.images = []
        self._point_at_map = {}

    @property
    def images_ids(self):
        return {image.image_id for image in self.images}

    def add_point(self, reconstructed_point: ReconstructedPoint) -> None:
        self.points[reconstructed_point.id] = reconstructed_point
        self._point_at_map[reconstructed_point.id] = len(self.points)

    def add_points(self, reconstructed_points: list[ReconstructedPoint]) -> None:
        for idx, reconstructed_point in enumerate(reconstructed_points, len(self.points)):
            self.points[reconstructed_point.id] = reconstructed_point
            self._point_at_map[reconstructed_point.id] = idx

    def visible_point_for(self, image: Image) -> list[ReconstructedPoint]:
        visible_points = []
        for inv_tracklet in self._scene_graph.keypoint_invtracks[image.image_id].values():
            reconstructed_point = self.points.get(inv_tracklet.idx)
            if reconstructed_point is not None:
                visible_points.append(reconstructed_point)

        return visible_points

    def visible_point_idxs(self, image: Image) -> list[int]:
        visible_point_idxs = []
        for inv_tracklet in self._scene_graph.keypoint_invtracks[image.image_id].values():
            idx = self._point_at_map.get(inv_tracklet.idx)
            if idx is not None:
                visible_point_idxs.append(idx)

        return visible_point_idxs

    def add_image(self, image: Image):
        self.images.append(image)

    def __contains__(self, item):
        image_id = item
        if isinstance(item, Image):
            image_id = item.image_id
        return image_id in self.images_ids
