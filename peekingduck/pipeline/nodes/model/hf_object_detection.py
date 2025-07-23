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

"""🔲 HuggingFace Auto Model for Object Detection."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.hf_object_detectionv1.auto_model import AutoModel


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses a HuggingFace transformer AutoModelforObjectDetection 
    to perform inference on an image frame.

    AutoModelforObjectDetection models are capable detecting objects from 80 categories. 
    The table of object categories can be found
    :ref:`here <general-object-detection-ids>`. The ``"PekingU/rtdetr_r50vd "`` model is
    used by default.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:

    References:
        Object Detection with HuggingFace Transformers:
        https://huggingface.co/docs/transformers/en/tasks/object_detection
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = AutoModel(self.config)


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
            "huggingface_model_path": str,
            "model_format": str,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
