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

"""Detector module to predict object bbox from an image using RT-DETR."""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class Detector:  # pylint: disable=too-many-instance-attributes
    """Object detection class using RE-DETR to predict object bboxes.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): RT-DETR node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        model (RTDetrForObjectDetection): The RT-DETR model for performing inference.
        image_processor (RTDetrImageProcessor): The RT-DETR image processor.
    """
    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_path: Union[Path, str],
        detect_ids: List[int],
        score_threshold: float=0.5,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.detect_ids = detect_ids
        self.score_threshold = score_threshold

        self.model, self.image_processor = self.create_rtdetr_model()

        self.id2label = self.model.config.id2label
        label2id = {v: k for k, v in self.id2label.items()}
        self.detect_ids = [label2id.get(i, 0) for i in self.detect_ids]

        self.log()


    @torch.no_grad()
    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detects bounding boxes of selected object categories from an image.

        The input image is first processed by the image processor.
        Detection results will be filtered according to `iou_threshold`, 
        `score_threshold`, and `detect_ids` configuration options. 
        Bounding boxes coordinates are then normalized w.r.t. the
        input `image` size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        # Store the original image size to normalize bbox later
        image_shape = image.shape[:2]

        inputs = self.preprocess(image)
        
        self.model = self.model.to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)

        bboxes, classes, scores = self.postprocess(result, image_shape)
        return bboxes, classes, scores


    def create_rtdetr_model(self) -> Tuple[RTDetrForObjectDetection, RTDetrImageProcessor]:
        """Creates a RT-DETR model and loads its weights. Also loads the image
        processor required to preprocess the input image and postprocess the 
        inference results.

        Returns:
            (RTDetrForObjectDetection): RT-DETR model.
            (RTDetrImageProcessor): RT-DETR image processor.
        """
        return (
            RTDetrForObjectDetection.from_pretrained(self.model_path),
            RTDetrImageProcessor.from_pretrained(self.model_path, use_fast=True),
        )


    def preprocess(self, image, return_tensors="pt"):
        # RT-DETR image processor takes images as RGB.
        # To tensor on CUDA for faster preprocessing.
        images = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device)
        inputs = self.image_processor(images=images, return_tensors=return_tensors)
        return inputs.to(self.device)
    

    def postprocess(self, predictions, image_shape):
        result = self.image_processor.post_process_object_detection(
            predictions, 
            target_sizes=torch.tensor([(image_shape[0], image_shape[1])]), 
            threshold=self.score_threshold,
        )[0]

        bboxes = result["boxes"].detach().cpu().numpy()
        classes = result["labels"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()

        want = np.isin(classes, self.detect_ids)
        bboxes = bboxes[want]
        classes = classes[want]
        scores = scores[want]

        bboxes = xyxy2xyxyn(bboxes, image_shape[0], image_shape[1])
        classes = np.array([self.id2label[c] for c in classes])
        return bboxes, classes, scores


    def log(self):
        self.logger.info(
            "RE-DETR model loaded with the following configs:\n\t"
            f"Model path: {self.model_path}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
        )
