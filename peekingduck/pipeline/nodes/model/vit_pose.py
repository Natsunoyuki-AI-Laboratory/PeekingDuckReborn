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

"""🔲 VITpose pose estimation model."""

from typing import Any, Dict, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.vit_posev1.vit_pose_model import VITPoseModel


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses VITPose to infer keypoints from an image frame.

    The VITPose node is capable of detecting keypoints from bounding boxes.
    The ``"usyd-community/vitpose-plus-small"`` model is used by default and can be 
    changed to one of ``("usyd-community/vitpose-plus-small", "usyd-community/vitpose-plus-base", 
    "usyd-community/vitpose-plus-large", "usyd-community/vitpose-plus-huge", 
    "usyd-community/vitpose-base", "usyd-community/vitpose-base-simple", 
    "usyd-community/vitpose-base-coco-aic-mpii")``.

    Inputs:
        |img_data|

        |bboxes_data|

    Outputs:
        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

    Configs:
        model_format (:obj:`str`): **{"pytorch"},
            default="pytorch"** |br|
            Defines the weights format of the model.
        model_path (:obj:`str`): **{"usyd-community/vitpose-plus-small", 
            "usyd-community/vitpose-plus-base",
            "usyd-community/vitpose-plus-large", 
            "usyd-community/vitpose-plus-huge", 
            "usyd-community/vitpose-base", 
            "usyd-community/vitpose-base-simple", 
            "usyd-community/vitpose-base-coco-aic-mpii"}, 
            default="usyd-community/vitpose-plus-small"**. |br|
            Defines the type of VITPose model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        keypoint_score_threshold (:obj:`float`): 
            **[0, 1], default = 0.5**. |br|
            Keypoints with confidence score below the threshold will be
            replaced by -1.
        online (:obj:`bool`): **default = True**. |br|
            If online == True, the weights from HuggingFace will be downloaded and used.
            If False, local weights will be loaded from weights_parent_dir by default.
            Pass a full path to a custom finetuned model to use those weights.

    References:
        ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation:
        https://huggingface.co/docs/transformers/v4.48.2/model_doc/vitpose

        Implementation on HuggingFace:
        https://huggingface.co/docs/transformers/v4.48.2/model_doc/vitpose

        Inference code and model weights:
        ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = VITPoseModel(self.config)


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the bbox input and returns the poses and pose bbox of the
        specified objects chosen to be detected.
        """
        keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"], inputs["bboxes"],
        )
        return {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
        }


    def _get_config_types(self) -> Dict[str, Any]:
        """
        Returns dictionary mapping the node's config keys to respective types.
        """
        return {
            "model_format": str,
            "model_path": str,
            "keypoint_score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
