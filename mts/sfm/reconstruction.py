from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mts.model.image import Image
from mts.model.point import Point3D
from mts.model.rgb import RGB
from mts.model.track import KeypointTrack
from mts.sfm.scene_graph import SceneGraph
from mts.types import Id


@dataclass
class ReconstructedPoint:
    id: Id = field(default=None)
    loc: Point3D = field(default=None)
    color: RGB = field(default=None)
    track: KeypointTrack = field(default=None)
    error: float = field(default=np.inf)
    scene_graph: SceneGraph = field(default=None, repr=False)

    def poses(self, not_none: bool = True):
        return [
            tracklet.image.pose
            for tracklet in self.track
            if not not_none or tracklet.image.pose is not None
        ]

    def kp_image(self, not_none: bool = True):
        return [
            tracklet.image_kp
            for tracklet in self.track
            if not not_none or tracklet.image.pose is not None
        ]

    def images(self, not_none: bool = True):
        return [
            tracklet.image
            for tracklet in self.track
            if not not_none or tracklet.image.pose is not None
        ]

    def kp_camera(self, not_none: bool = True, homogenous: bool = False):
        homogenous_slice = slice(None, -1)
        if homogenous:
            homogenous_slice = slice(None, 2)
        if not_none:
            return [
                tracklet.camera_kp[homogenous_slice]
                for tracklet in self.track
                if tracklet.image.pose is not None
            ]
        if homogenous:
            return [tracklet.camera_kp[homogenous_slice] for tracklet in self.track]


@dataclass
class Reconstruction:

    images: List[Image] = field(default_factory=list)
    points: Dict[int, ReconstructedPoint] = field(default=None)

    @property
    def images_ids(self):
        return {image.image_id for image in self.images}

    def add_image(self, image: Image):
        self.images.append(image)

    def __contains__(self, item):
        image_id = item
        if isinstance(item, Image):
            image_id = item.image_id
        return image_id in self.images_ids
