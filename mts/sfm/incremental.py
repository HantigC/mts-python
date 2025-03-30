from __future__ import annotations
from dataclasses import dataclass, field
import logging
import random
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from mts.estimator.pnp.dlt import LinearPNPRansac
from mts.geometry.triangulation import linear
from mts.keypoint.base import BaseMatcher
from mts.model.image import Image, ImageId, ImageType
from mts.model.point import Point3D
from mts.model.rgb import RGB
from mts.model.two_view import PairId, TwoViewPair, compute_two_view
from mts.pose.rigid import Rigid3D
from mts.sfm.reconstruction import Reconstruction
from mts.sfm.scene_graph import SceneGraph, View3DPair
from mts.types import NPVector3f

LOGGER = logging.getLogger(__name__)


StartingPairType = Union[
    TwoViewPair,
    Tuple[ImageType, ImageType],
    PairId,
]


@dataclass
class IncrementalSfMConfig:
    min_init_inliers: int = field(default=50)
    min_pose_inliers: int = field(default=50)
    min_angle: float = field(default=15)
    min_reprojection_error: float = field(default=2)
    max_depth: float = field(default=20)
    min_depth: float = field(default=1)


class IncrementalSfM:

    def __init__(
        self,
        images_map: Dict[ImageId, Image],
        image_pair_ids: List[Tuple[ImageId, ImageId]],
        pnp_estimator: LinearPNPRansac,
        matcher: BaseMatcher = None,
        min_init_inliers: int = 50,
        min_pose_inliers: int = 50,
        min_angle: float = 15,
        min_reprojection_error: float = 2,
        max_depth: float = 20,
        min_depth: float = 1,
    ) -> None:
        self.images_map = images_map
        self.image_pair_ids = image_pair_ids
        self.scene_graph = None

        self.two_view_map = {}
        self.pairs_map = {}
        self.min_angle = min_angle
        self.sorted_twoview_pairs = []
        self.pnp_estimator = pnp_estimator

        self.min_init_inliers = min_init_inliers
        self.min_pose_inliers = min_pose_inliers
        self.max_reprojection_error = min_reprojection_error
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.matcher = matcher
        self.reconstructions = []

    def _compute_scene_graph(self):
        self.scene_graph = SceneGraph(self.images_map)

        for st_image_id, nd_image_id in self.image_pair_ids:
            two_view = compute_two_view(
                self.matcher.match,
                self.images_map[st_image_id],
                self.images_map[nd_image_id],
            )
            if two_view is None or len(two_view.matches) < self.min_init_inliers:

                continue
            self.scene_graph.add_two_view_pair(two_view)

    def _check_if_id(self, image_id: ImageId, two_view_pair: TwoViewPair) -> bool:
        return image_id in two_view_pair.image_pair_ids

    def _check_if_ids(
        self,
        two_view_pair: TwoViewPair,
        image_ids: List[ImageId],
    ) -> bool:
        return any(
            self._check_if_id(
                image_id,
                two_view_pair,
            )
            for image_id in image_ids
        )

    def _check_depth_interval(self, pose: Rigid3D, point3d: NPVector3f) -> bool:
        return self.min_depth <= pose.z_of(point3d) <= self.max_depth

    def init_3d_points(self, two_view_pair: TwoViewPair) -> None:

        st_image = two_view_pair.st_image
        nd_image = two_view_pair.nd_image
        points = {}

        for (st_match, nd_match), (x, y, z) in zip(
            two_view_pair.matches, two_view_pair.points3D
        ):
            st_track_idx = self.scene_graph.keypoint_invtracks[st_image.image_id][
                st_match
            ].idx
            nd_track_idx = self.scene_graph.keypoint_invtracks[nd_image.image_id][
                nd_match
            ].idx
            if st_track_idx != nd_track_idx:
                LOGGER.warning(
                    "(%d, %d) matches do not point to the same track %d~ != %d",
                    st_match,
                    nd_match,
                    st_track_idx,
                    nd_track_idx,
                )
                continue
            if not (
                self._check_depth_interval(two_view_pair.relative_pose, [x, y, z])
                and self._check_depth_interval(Rigid3D.from_identity(), [x, y, z])
            ):
                continue

            self.scene_graph.points[st_track_idx].loc = Point3D(x, y, z)
            i, j = st_image.keypoints[st_match]
            self.scene_graph.points[st_track_idx].color = RGB(
                *st_image.img[int(j), int(i)]
            )

            points[st_track_idx] = self.scene_graph.points[st_track_idx]
        return points

    def _start_from_pair(self, pair: TwoViewPair) -> None:
        self.scene_graph.images[pair.st_image.image_id].pose = Rigid3D.from_identity()
        self.scene_graph.images[pair.nd_image.image_id].pose = pair.relative_pose
        initial_3d_points = self.init_3d_points(pair)

        reconstruction = Reconstruction(
            images=[
                pair.st_image,
                pair.nd_image,
            ],
            points=initial_3d_points,
        )
        LOGGER.info(
            "Starting from pair: (%d, %d) with angle (%f)",
            pair.st_image.image_id,
            pair.nd_image.image_id,
            pair.angle,
        )

        self.reconstructions.append(reconstruction)

    def _prepare_starting_pair(
        self,
        start_pair: Optional[StartingPairType] = None,
    ) -> TwoViewPair:
        if start_pair is None:
            return None

        if isinstance(start_pair, TwoViewPair):
            return start_pair
        elif isinstance(start_pair, tuple):
            return self.scene_graph.pair_at(*start_pair)
        elif isinstance(start_pair, PairId):
            return self.scene_graph.pairs[start_pair]
        raise ValueError(f"starting pair should be a type of {StartingPairType}")

    def _compute_starting_pair(self, image_ids: List[ImageId]) -> TwoViewPair:
        starting_pair = None
        for pair in sorted(
            self.scene_graph.pairs.values(),
            key=lambda pair: len(pair.matches),
            reverse=True,
        ):
            if not self._check_if_ids(pair, image_ids):
                continue

            if len(pair.matches) < self.min_init_inliers:
                continue

            if pair.angle < self.min_angle:
                continue

            starting_pair = pair
            break
        return starting_pair

    def _generate_next_image(
        self,
        reconstruction: Reconstruction,
        image_ids: List[ImageId],
    ) -> Generator[Image, None, None]:
        possible_images = [
            image
            for image in self.scene_graph.images.values()
            if image not in reconstruction and image.image_id in image_ids
        ]
        sorted_images = sorted(
            possible_images,
            key=lambda image: len(self.scene_graph.visible_pair_for(image)),
            reverse=True,
        )
        for image in sorted_images:
            yield image, self.scene_graph.visible_pair_for(image)

    def _get_visible_points(
        self,
        image: Image,
        reconstruction: Reconstruction,
    ) -> View3DPair:
        matcher_cv = cv2.BFMatcher()
        descriptors_list = []
        for train_image in reconstruction.images:
            descriptors_list.append(train_image.descriptors)

        matcher_cv.add(descriptors_list)

        matches = matcher_cv.match(image.descriptors)

        kps = []
        points = []
        processed_points = set()
        for match in matches:
            matched_image = reconstruction.images[match.imgIdx]
            inv_track = self.scene_graph.keypoint_invtracks[matched_image.image_id].get(
                match.trainIdx
            )
            if inv_track is not None:
                if self.scene_graph.points[inv_track.idx].loc is not None:
                    points.append(self.scene_graph.points[inv_track.idx].loc.as_np())
                    processed_points.add(inv_track.idx)
                    kps.append(image.keypoints[match.queryIdx])

        kps = np.stack(kps)
        points = np.stack(points)
        return View3DPair(points, kps)

    def _compute_new_pose(
        self,
        image: Image,
        reconstruction: Reconstruction,
    ) -> Tuple[Rigid3D, np.ndarray, View3DPair]:
        next_points = self.scene_graph.visible_pair_for(image, False)
        # match_next_points = self._get_visible_points(image, reconstruction)
        # if len(match_next_points) > len(next_points):
        #     next_points = match_next_points
        pose = None
        mask = None

        if len(next_points) == 0:
            return pose, mask, None

        pnp_summary = self.pnp_estimator.estimate(
            next_points.world_points,
            next_points.camera_points,
            image.K,
        )

        if (
            pnp_summary is not None
            and sum(pnp_summary.mask) > self.min_pose_inliers
            and np.mean(pnp_summary.projection_errors) < self.max_reprojection_error
        ):
            pose = Rigid3D(pnp_summary.R, pnp_summary.t)
            mask = pnp_summary.mask
            LOGGER.info(
                "Compute new pose for image %d with reprojection error %f",
                image.image_id,
                np.mean(pnp_summary.projection_errors),
            )
        return pose, mask, next_points

    def _register_new_image(
        self,
        image: Image,
        pose: Rigid3D,
        reconstruction: Reconstruction,
    ) -> None:

        image.pose = pose
        reconstruction.images.append(image)
        added_new_points = 0
        for kp_idx, inv_track in self.scene_graph.keypoint_invtracks[
            image.image_id
        ].items():
            poses = []
            points = []
            reconstructed_point = self.scene_graph.points[inv_track.idx]
            if reconstructed_point.loc is not None:
                continue

            i, j = image.keypoints[kp_idx]
            posed_images = []
            for tracklet in reconstructed_point.track:
                tracked_image = self.scene_graph.images[tracklet.image.image_id]
                pose = tracked_image.pose
                if pose is not None:
                    poses.append(pose.Rt4x4)
                    posed_images.append(tracked_image)
                    points.append(tracked_image.camera_keypoints[tracklet.keypoint_num])

            if len(poses) >= 2:
                # point_3d = linear.multi_cameras(points, poses)
                point_3d = self.triangulate_points(points, poses, reconstructed_point)

                error = self.scene_graph.projection_error_for(
                    point_3d, reconstructed_point
                )
                if error > self.max_reprojection_error:
                    continue
                if error > reconstructed_point.error:
                    continue

                if any(
                    depth > self.max_depth or depth < self.min_depth
                    for depth in self.scene_graph.depths_for(
                        point_3d, reconstructed_point
                    )
                ):
                    continue

                reconstructed_point.loc = Point3D.from_numpy(point_3d)
                reconstructed_point.error = error
                reconstructed_point.color = RGB(*image.img[int(j), int(i)])
                if inv_track.idx not in reconstruction.points:
                    reconstruction.points[inv_track.idx] = reconstructed_point
                    added_new_points += 1

        LOGGER.info("Add new %d points", added_new_points)

    def triangulate_points(self, points, poses, reconstructed_point, trials=10):
        best_point = None
        best_error = np.inf
        for _ in range(trials):
            st_idx, nd_idx = np.random.randint(0, len(points), 2)
            st_points, nd_points = points[st_idx], points[nd_idx]
            st_pose, nd_pose = poses[st_idx], poses[nd_idx]
            point_3d = linear.two_cameras([st_points], [nd_points], st_pose, nd_pose)

            error = self.scene_graph.projection_error_for(
                point_3d[0], reconstructed_point
            )
            if error < best_error:
                best_point = point_3d[0]
                best_error = error
        return best_point

    def _get_best_image(
        self,
        reconstruction: Reconstruction,
        image_ids: List[ImageId],
    ) -> Tuple[Image, Rigid3D, View3DPair]:
        best_image = None
        best_inliers = -1
        best_pose = None
        best_visible_points = None
        for next_image, _ in self._generate_next_image(reconstruction, image_ids):
            if next_image is None:
                break
            pose, mask, visible_points = self._compute_new_pose(
                next_image, reconstruction
            )
            if pose is None:
                continue
            LOGGER.info(
                f"Trying the image `{next_image.image_id}` with `{np.sum(mask): 02d}` inliers",
            )
            if sum(mask) > best_inliers:
                best_inliers = sum(mask)
                best_image = next_image
                best_visible_points = visible_points[mask]
                best_pose = pose
        return best_image, best_pose, best_visible_points

    def build(
        self,
        image_ids=None,
        start_pair: Optional[StartingPairType] = None,
    ) -> None:
        self._compute_scene_graph()
        image_ids = image_ids or self.scene_graph.image_ids
        start_pair = self._prepare_starting_pair(
            start_pair
        ) or self._compute_starting_pair(image_ids)

        self._start_from_pair(start_pair)
        while True:
            # next_image = self.scene_graph.images[10]
            # visible_points = self.scene_graph.visible_pair_for(next_image)
            next_best_image, next_best_pose, visible_points = self._get_best_image(
                self.reconstructions[-1],
                image_ids,
            )
            if next_best_image is None:
                break

            LOGGER.info(
                "Registering image %d with %d visible points",
                next_best_image.image_id,
                len(visible_points),
            )
            self._register_new_image(
                next_best_image,
                next_best_pose,
                self.reconstructions[-1],
            )

    @classmethod
    def from_config(
        cls,
        images_map: Dict[ImageId, Image],
        image_pair_ids: List[Tuple[ImageId, ImageId]],
        pnp_estimator: LinearPNPRansac,
        config: IncrementalSfMConfig,
        matcher: BaseMatcher = None,
    ) -> IncrementalSfM:
        return cls(
            images_map,
            image_pair_ids,
            pnp_estimator,
            matcher=matcher,
            **config.__dict__,
        )
