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

"""🔲 RT-DETR real time transformer object detector."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.rt_detrv1.rt_detr_model import RTDETRModel


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses RT-DETR to infer from an image frame.

    The RT-DETR node is capable detecting objects from 80 categories. The table
    of object categories can be found
    :ref:`here <general-object-detection-ids>`. The ``"PekingU/rtdetr_r50vd "`` model is
    used by default and can be changed to one of ``("PekingU/rtdetr_r18vd", 
    "PekingU/rtdetr_r34vd", "PekingU/rtdetr_r50vd", "PekingU/rtdetr_r101vd", 
    "PekingU/rtdetr_r18vd_coco_o365", "PekingU/rtdetr_r50vd_coco_o365", 
    "PekingU/rtdetr_r101vd_coco_o365")``.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:
        model_format (:obj:`str`): **{"pytorch"},
            default="pytorch"** |br|
            Defines the weights format of the model.
        model_path (:obj:`str`): **{"PekingU/rtdetr_r18vd", "PekingU/rtdetr_r34vd", 
            "PekingU/rtdetr_r50vd", "PekingU/rtdetr_r101vd", "PekingU/rtdetr_r18vd_coco_o365",
            "PekingU/rtdetr_r50vd_coco_o365", "PekingU/rtdetr_r101vd_coco_o365"}, 
            default="PekingU/rtdetr_r50vd"**. |br|
            Defines the type of RT-DETR model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        input_size (:obj:`int`): **default=640**. |br|
            Input image resolution of the RT-DETR model.
        detect (:obj:`List[Union[int, str]]`): **default=[0]**. |br|
            List of object class names or IDs to be detected. To detect all classes,
            refer to the :ref:`tech note <general-object-detection-ids>`.
        score_threshold (:obj:`float`): **[0, 1], default = 0.25**. |br|
            Bounding boxes with confidence score (product of objectness score
            and classification score) below the threshold will be discarded.

    References:
        DETRs Beat YOLOs on Real-time Object Detection:
        https://arxiv.org/pdf/2304.08069

        RT-DETR implementation on HuggingFace:
        https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr

        Inference code and model weights:
        https://github.com/lyuwenyu/RT-DETR
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = RTDETRModel(self.config)


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes of the detect
        objects.

        The classes of objects to be detected can be specified through the
        `detect` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        bboxes, labels, scores = self.model.predict(inputs["img"])
        bboxes = np.clip(bboxes, 0, 1)
        return {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}


    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "detect": List[Union[int, str]],
            "input_size": int,
            "model_format": str,
            "model_path": str,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
