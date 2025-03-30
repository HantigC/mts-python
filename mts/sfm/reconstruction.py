
from dataclasses import dataclass, field
from typing import Dict, List

from mts.model.image import Image
from mts.sfm.scene_graph import ReconstructedPoint


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