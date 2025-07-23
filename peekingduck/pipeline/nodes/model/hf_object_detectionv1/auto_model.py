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

"""RT-DETR models with model types: rtdetr_r18vd, rtdetr_r34vd, rtdetr_r50vd,
rtdetr_r101vd, rtdetr_r18vd_coco_o365, rtdetr_r50vd_coco_o365, 
rtdetr_r101vd_coco_o365."""

import logging
from typing import Any, Dict, List, Tuple

from pathlib import Path
import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.hf_object_detectionv1.auto_model_files.detector import Detector


class AutoModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["score_threshold"], "[0, 1]")

        local_weights_path = self.config.get("local_weights_path", None)
        if local_weights_path is None:
            if self.config["online"] is True:
                # Download the weights from HuggingFace in online mode.
                model_path = self.config["huggingface_model_path"]
            else:
                # Load HuggingFace pre-trained weights from peekingduck_weights.
                model_dir = self._find_paths()
                model_path = self.config["huggingface_model_path"]
                model_path = model_dir / model_path
        else:
            # Absolute path to custom local weights directory.
            model_path = Path(local_weights_path)

        self.detect_ids = self.config["detect"]

        self.detector = Detector(
            model_path,
            self.detect_ids,
            self.config["input_size"],
            self.config["score_threshold"],
        )


    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids


    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = ids


    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image)
