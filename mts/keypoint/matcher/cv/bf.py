import cv2
import numpy as np
from mts.keypoint.base import BaseMatcher


class BFMatcher(BaseMatcher):
    def __init__(self, threshold: float = 0.75) -> None:
        self.bf = cv2.BFMatcher()
        self.threshold = threshold

    def match(self, st_descriptors, nd_descriptors):
        good_matches = []
        matches = self.bf.knnMatch(st_descriptors, nd_descriptors, 2)
        matches = sorted(matches, key=lambda m: m[0].distance)
        query_set = set()
        train_set = set()

        for m, n in matches:
            if m.queryIdx in query_set or m.trainIdx in train_set:
                continue

            if m.distance < self.threshold * n.distance:
                query_set.add(m.queryIdx)
                train_set.add(m.trainIdx)
                good_matches.append((m.queryIdx, m.trainIdx))

        good_matches = np.array(good_matches)
        return good_matches
