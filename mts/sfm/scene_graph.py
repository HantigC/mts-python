from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from mts.estimator.pnp.dlt import ComputeReprojectionError
from mts.model.image import Image, ImageType, ImageId
from mts.model.point import Point3D
from mts.model.rgb import RGB
from mts.pose.rigid import compute_depth_mask
from mts.model.track import (
    InvTrack,
    KeypointInvTrack,
    KeypointTrack,
    TrackLocation,
    intersect_tracks,
)
from mts.model.two_view import TwoViewPair, image_ids_to_pair_id, PairId
from mts.types import Id, NPVector3f


LOGGER = logging.getLogger("SfM")


@dataclass
class ReconstructedPoint:
    id: Id = field(default=None)
    loc: Point3D = field(default=None)
    color: RGB = field(default=None)
    track: KeypointTrack = field(default=None)
    error: float = field(default=np.inf)


class View3DPair(NamedTuple):
    world_points: np.ndarray
    camera_points: np.ndarray

    def __len__(self):
        return len(self.world_points)

    def __getitem__(self, index):
        return View3DPair(self.world_points[index], self.camera_points[index])


@dataclass
class SceneGraph:
    images: Dict[ImageId, Image]
    pairs: Optional[Dict[PairId, TwoViewPair]] = field(default_factory=dict)

    pairs_map: Dict[ImageId, List[TwoViewPair]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )
    points: Dict[int, ReconstructedPoint] = field(init=False, default_factory=dict)

    keypoint_invtracks: Dict[ImageId, KeypointInvTrack] = field(
        init=False,
        default_factory=dict,
    )
    _last_track_id: int = field(default=-1, init=False)

    def __post_init__(self):
        for pair in self.pairs.values():
            self.pairs_map[pair.st_image.image_id].append(pair)
            self.pairs_map[pair.nd_image.image_id].append(pair)

    def _extract_image_id(self, image_id: ImageType) -> ImageId:
        if not isinstance(image_id, ImageId):
            if not isinstance(image_id, Image):
                raise ValueError(
                    f"image should be of type {ImageType}, not {type(image_id)}",
                )
            image_id = image_id.image_id

        return image_id

    def _get_image(self, image: ImageType) -> Image:
        if isinstance(image, Image):
            return image

        if not isinstance(image, ImageId):
            raise ValueError(
                f"image should be of type {ImageType}, not {type(image)}",
            )
        return self.images[image]

    def pair_at(
        self,
        st_image: ImageType,
        nd_image: ImageType,
    ) -> TwoViewPair:
        st_image_id = self._extract_image_id(st_image)
        nd_image_id = self._extract_image_id(nd_image)

        pair_id = image_ids_to_pair_id(st_image_id, nd_image_id)
        two_view_pair = self.pairs[pair_id]
        return two_view_pair

    @property
    def image_ids(self):
        return self.images.keys()

    @property
    def pair_ids(self):
        return self.pairs.keys()

    def __contains__(self, item: Image | TwoViewPair) -> bool:
        if isinstance(item, Image):
            return item.image_id in self.images
        elif isinstance(item, TwoViewPair):
            return item.pair_id in self.pairs
        else:
            raise ValueError(f"`{type(item)}` is not contained in the scene graph")

    def pairs_for(self, image_id: ImageType) -> List[TwoViewPair]:
        image_id = self._extract_image_id(image_id)
        return self.pairs_map[image_id]

    def add_two_view_pair(
        self, two_view_pair: TwoViewPair, strict: bool = True
    ) -> None:
        if two_view_pair.pair_id in self.pairs and strict:
            raise ValueError(
                f"pair id `{two_view_pair.pair_id}` already exists in the scene graph"
            )
        self.pairs[two_view_pair.pair_id] = two_view_pair
        self.pairs_map[two_view_pair.st_image.image_id].append(two_view_pair)
        self.pairs_map[two_view_pair.nd_image.image_id].append(two_view_pair)
        self._add_tracks(two_view_pair)

    def add_image_(self, image: Image) -> None:
        if image.image_id in self.image_ids:
            raise ValueError(
                f"image id `{image.image}` already exists in the scene graph"
            )
        self.images[image] = image

    def visible_points_for(self, images: List[ImageId]):
        images = [self._get_image(image) for image in images]
        images = sorted(images, key=len(self.keypoint_invtracks[st_image.image_id]))

        st_image = self._get_image(image[0])

        inv_tracks = self.keypoint_invtracks[st_image.image_id]
        for image in images[1:]:
            intersect_tracks(inv_tracks, self.keypoint_invtracks[image.image_id])
        # TODO: implement a method that extracts reconstructed points for inv tracks

    def visible_pair_for(
        self, image: ImageId | Image, in_camera: bool = True
    ) -> View3DPair:
        if not isinstance(image, Image):
            image = self.images[image]

        world_points = []
        camera_points = []
        if image.image_id not in self.keypoint_invtracks:
            return View3DPair(world_points, camera_points)

        kp_getter = None
        if in_camera:
            kp_getter = image.camera_keypoints
        else:
            kp_getter = image.keypoints
        for kp_idx, inv_track in self.keypoint_invtracks[image.image_id].items():
            reconstructed_point = self.points[inv_track.idx]
            if reconstructed_point.loc is not None:
                camera_points.append(kp_getter[kp_idx])
                world_points.append(reconstructed_point.loc.as_np())
        world_points = np.stack(world_points)
        camera_points = np.stack(camera_points)
        return View3DPair(world_points, camera_points)

    def depths_for(
        self,
        point_3d: NPVector3f,
        reconstructed_point: ReconstructedPoint,
    ) -> List[float]:
        depths = []

        for tracklet in reconstructed_point.track:
            tracked_image: Image = self.images[tracklet.image.image_id]
            if tracked_image.pose is None:
                continue
            depths.append(tracked_image.pose.z_of(point_3d))
        return depths

    def projection_error_for(
        self,
        point_3d: NPVector3f,
        reconstructed_point: ReconstructedPoint,
    ) -> float:
        errors = []
        for tracklet in reconstructed_point.track:
            tracked_image: Image = self.images[tracklet.image.image_id]
            tracked_image.keypoints[tracklet.keypoint_num]
            if tracked_image.pose is None:
                continue
            point_2d = tracked_image.project(point_3d)
            errors.append(ComputeReprojectionError(point_2d, tracklet.image_kp))
        return np.mean(errors)

    def _add_tracks(
        self,
        two_view_pair: TwoViewPair,
    ) -> None:
        st_image = two_view_pair.st_image
        nd_image = two_view_pair.nd_image
        st_kp_mask: KeypointInvTrack = self.keypoint_invtracks.setdefault(
            st_image.image_id, {}
        )
        nd_kp_mask: KeypointInvTrack = self.keypoint_invtracks.setdefault(
            nd_image.image_id, {}
        )
        for st_match, nd_match in two_view_pair.matches:
            st_inv_track = st_kp_mask.get(st_match)
            nd_inv_track = nd_kp_mask.get(nd_match)
            if st_inv_track is None and nd_inv_track is None:
                self._create_track(
                    st_image,
                    nd_image,
                    st_match,
                    nd_match,
                    st_kp_mask,
                    nd_kp_mask,
                )

            elif st_inv_track is None:
                last_inv_loc = len(self.points[nd_inv_track.idx].track)
                st_kp_mask[st_match] = InvTrack(nd_inv_track.idx, last_inv_loc)
                self.points[nd_inv_track.idx].track.append(
                    TrackLocation(st_image, st_match)
                )

            elif nd_inv_track is None:
                last_inv_loc = len(self.points[st_inv_track.idx].track)
                nd_kp_mask[nd_match] = InvTrack(st_inv_track.idx, last_inv_loc)
                self.points[st_inv_track.idx].track.append(
                    TrackLocation(nd_image, nd_match)
                )
            else:
                if st_inv_track.idx == nd_inv_track.idx:
                    continue
                self._merge_tracks(st_inv_track, nd_inv_track)

    def _create_track(
        self,
        st_image: Image,
        nd_image: Image,
        st_match: int,
        nd_match: int,
        st_kp_mask,
        nd_kp_mask,
    ) -> None:
        self._last_track_id += 1

        reconstructed_point = ReconstructedPoint(
            id=self._last_track_id,
            track=[
                TrackLocation(st_image, st_match),
                TrackLocation(nd_image, nd_match),
            ],
        )
        self.points[self._last_track_id] = reconstructed_point
        st_kp_mask[st_match] = InvTrack(self._last_track_id, 0)
        nd_kp_mask[nd_match] = InvTrack(self._last_track_id, 1)

    def _merge_tracks(self, st_inv_track: InvTrack, nd_inv_track: InvTrack) -> None:
        reconstructed_point = self.points.pop(nd_inv_track.idx)
        starting_len = len(self.points[st_inv_track.idx].track)
        for num, tracklet in enumerate(reconstructed_point.track, starting_len):
            self.keypoint_invtracks[tracklet.image.image_id][tracklet.keypoint_num] = (
                InvTrack(st_inv_track.idx, num)
            )
            self.points[st_inv_track.idx].track.append(tracklet)

    @classmethod
    def from_images_(cls, images: Sequence[Image]):
        for image_id, image in enumerate(images):
            image.image_id = image_id
        return cls({image.image_id: image for image in images})
