from dataclasses import dataclass
from typing import Dict, List, NamedTuple, TypeAlias

from mts.model.image import Image
from mts.model.point import Point3D


class TrackLocation(NamedTuple):
    image: Image
    keypoint_num: int

    @property
    def image_kp(self):
        return self.image.keypoints[self.keypoint_num]

    @property
    def camera_kp(self):
        return self.image.camera_keypoints[self.keypoint_num]

    def __repr__(self):
        return f"{self.__class__.__name__}(image={self.image.image_id}, keypoint_num={self.keypoint_num},)"


@dataclass
class PointTrack:
    track: List[TrackLocation]
    point: Point3D = None


class InvTrack(NamedTuple):
    idx: int
    nth: int


def intersect_tracks(
    st_invtracks_sorted: List[InvTrack],
    nd_invtracks_sorted: List[InvTrack],
) -> List[InvTrack]:
    st_cnt = 0
    nd_cnt = 0
    intersected_inv_tracks = []
    try:
        while True:
            st_inv_track = st_invtracks_sorted[st_cnt]
            nd_inv_track = nd_invtracks_sorted[nd_cnt]
            st_idx = st_inv_track.idx
            nd_idx = nd_inv_track.idx
            if st_idx == nd_idx:
                intersected_inv_tracks.append(st_inv_track)
                st_cnt += 1
                nd_cnt += 1
            elif st_idx > nd_idx:
                nd_cnt += 1
            else:
                st_cnt += 1
    except IndexError:
        pass
    return intersected_inv_tracks


KeypointTrack: TypeAlias = List[TrackLocation]
KeypointInvTrack: TypeAlias = Dict[int, InvTrack]
