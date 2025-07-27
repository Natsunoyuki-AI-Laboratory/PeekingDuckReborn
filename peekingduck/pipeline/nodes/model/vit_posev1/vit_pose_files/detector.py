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

"""Detector module to predict pose keypoints from an image using VITPose."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

from peekingduck.pipeline.utils.bbox.transforms import xyxyn2tlwh


# MSCOCO has 17 keypoints.
N_KEYPOINTS = 17
KEYPOINT_LABELS = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7],
]


class Detector:  # pylint: disable=too-many-instance-attributes
    """Pose estimation class using VITPose to predict keypoints.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): VITPose node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        model (VitPoseForPoseEstimation): The VITPose model for performing 
            inference.
        image_processor (VitPoseImageProcessor): The VITPose image processor.
    """
    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_path: Union[str, Path],
        keypoint_score_threshold: float=0.5,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        
        self.keypoint_score_threshold = keypoint_score_threshold

        self.model, self.image_processor = self.create_model()

        self.log()


    def create_model(
        self) -> Tuple[VitPoseForPoseEstimation, VitPoseImageProcessor]:
        """Creates a VITPose model and loads its weights. Also loads the image
        processor required to preprocess and postprocess the inference results.

        Returns:
            (VitPoseForPoseEstimation): VITPose model.
            (VitPoseImageProcessor): VITPose image processor.
        """
        return (
            VitPoseForPoseEstimation.from_pretrained(self.model_path),
            VitPoseImageProcessor.from_pretrained(self.model_path),
        )


    @torch.no_grad()
    def predict_keypoints_from_image(
        self, image: np.ndarray, bboxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Detects keypoints corresponding to detected human bounding boxes in 
        an image frame.

        Args:
            image (np.ndarray): Input image.
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
        """
        # Store the original image size to normalize bbox later
        image_shape = image.shape[:2]

        if len(bboxes) > 0:
            bboxes = xyxyn2tlwh(
                bboxes, height=image_shape[0], width=image_shape[1]
            )
            inputs = self.preprocess(image, bboxes=bboxes)
            result = self.forward(inputs=inputs)
            keypoints, keypoint_scores, keypoint_conns = self.postprocess(
                result, bboxes, image_shape
            )
        else:
            keypoints = np.zeros(0)
            keypoint_scores = np.zeros(0)
            keypoint_conns = np.zeros(0)

        return keypoints, keypoint_scores, keypoint_conns


    def forward(self, inputs):
        """Forward pass with the model."""
        self.model = self.model.to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)
        return result


    def preprocess(self, image, bboxes, return_tensors="pt"):
        """Preprocess input images for ingestion by VITPose."""
        # HuggingFace image processors take in PIL images typically...
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.image_processor(
            images=image, boxes=[bboxes], return_tensors=return_tensors,
        ).to(self.device)
        # VITPose uses an MOE architecture. Need to specify dataset indexes for
        # each image.
        inputs["dataset_index"] = torch.tensor([0], device=self.device)
        return inputs.to(self.device)
    

    def postprocess(self, predictions, bboxes, image_shape):
        """Postprocess VITPose predictions into a format compatible with PKDR's
        data pipeline."""
        result = self.image_processor.post_process_pose_estimation(
            predictions, boxes=[bboxes],
        )[0]

        N_result = len(result)
        keypoints = np.zeros([N_result, N_KEYPOINTS, 2])
        keypoint_scores = np.zeros([N_result, N_KEYPOINTS])
        keypoint_conns = []
        for i, r in enumerate(result):
            # VITPose will only output detected keypoints. "Missing" keypoints
            # must be filled in for completeness of the 17 MSCOCO keypoints.
            kpts = r.get("keypoints").detach().cpu().numpy()
            kpt_scores = r.get("scores").detach().cpu().numpy()
            kpt_labels = r.get("labels").detach().cpu().numpy()
            
            # Normalize keypoints coordinates.
            kpts[:, 0] = kpts[:, 0] / image_shape[1]
            kpts[:, 1] = kpts[:, 1] / image_shape[0]

            # Fill in the missing MSCOCO keypoints. There are 17 in total.
            missing_kpts = np.array(
                list(set(KEYPOINT_LABELS).difference(set(kpt_labels)))
            )
            N_missing = len(missing_kpts)
            kpt_labels = np.concat([kpt_labels, missing_kpts])
            kpt_scores = np.concat([kpt_scores, np.array([0] * N_missing)])
            kpts = np.concat(
                [kpts, np.array([-1, -1] * N_missing).reshape(N_missing, 2)], 
                axis=0
            )
            
            # Sort all 17 keypoints by kpt_labels from 0 to 16.
            want = np.argsort(kpt_labels)
            kpt_scores = kpt_scores[want]
            kpts = kpts[want]
            
            # Mask out low confidence keypoints by -1.
            mask = kpt_scores >= self.keypoint_score_threshold
            kpts[np.logical_not(mask)] = -1

            keypoints[i, :, :] = kpts
            keypoint_scores[i, :] = kpt_scores
            keypoint_conns.append(get_mscoco_keypoint_connections(kpts, mask))

        return keypoints, keypoint_scores, keypoint_conns


    def log(self):
        self.logger.info(
            "VITPose model loaded with the following configs:\n\t"
            f"Model path: {self.model_path}\n\t"
            f"Keypoint score threshold: {self.keypoint_score_threshold}\n\t"
        )


def get_mscoco_keypoint_connections(
    keypoint: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Builds the keypoint connections between valid keypoints."""
    connections = []
    for start_joint, end_joint in SKELETON:
        if mask[start_joint - 1] and mask[end_joint - 1]:
            connections.append(
                (keypoint[start_joint - 1], keypoint[end_joint - 1])
            )

    return np.array(connections)
