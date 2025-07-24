# Copyright 2025 Natsunoyuki AI Laboratory
#
# PeekingDuckReborn is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
#
# PeekingDuckReborn is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# PeekingDuckReborn. If not, see <https://www.gnu.org/licenses/>.

import logging
from typing import Any, Dict, List, Tuple

from pathlib import Path
import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.vit_posev1.vit_pose_files.detector import Detector


class VITPoseModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Validates configuration, loads VITPose model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.

    Attributes:
        detector (Detector): VITPose keypoint detector object to infer human
            pose keypoints from a provided image frame.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["keypoint_score_threshold"], "[0, 1]")

        model_path = self.config["model_path"]
        if not Path(model_path).exists():
            if not self.config["online"]:
                model_dir = self._find_paths()
                model_path = model_dir / model_path

        self.detector = Detector(
            model_path,
            self.config["resolution"],
            self.config["keypoint_score_threshold"],
        )


    def predict(
        self, 
        image: np.ndarray,
        bboxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Predicts keypoints from image and bboxes.

        Args:
            image (np.ndarray): Input image frame.
            bboxes (np.ndarray): Person bounding boxes from an object detector.

        Returns:
            (Tuple[np.ndarray, np.ndarray, list]): Returned tuple
            contains:
            - An array of keypoints with shape [N, K, 2],
            - An array of keypoint scores with shape [N, K],
            - A list of length N of keypoint connections with shape [Dk, 2, 2],
            where N is the number of persons, K is the number of keypoints which
            is enforced to 17 following MSCOCO, and Dk is the number of valid
            keypoint connections.

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        return self.detector.predict_keypoints_from_image(image, bboxes)
