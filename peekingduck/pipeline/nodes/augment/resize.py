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

from typing import Any, Dict, Optional

import cv2

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin


class Node(ThresholdCheckerMixin, AbstractNode):
    """Resizes an input frame.

    Inputs:
        |img_data|

    Outputs:
        |img_data|

    Configs:
        max_dim (:obj:`int`): **[1, +inf), default = 640**. |br|
            New max dim of the resized image frame. The image will
            be resized such that the new largest dimension = max_dim.
            Aspect ratio is preserved.
        width (:obj:`Optional[int]`): **[1, +inf), default = null**. |br|
            Optional. New width of the resized image frame. 
            If None, max_dim will be used. 
            If both width and height are not None, max_dim will be ignored.
            Aspect ratio might not be preserved in this case.
        height (:obj:`Optional[int]`): **[1, +inf), default = None**. |br|
            Optional. New height of the resized image frame. 
            If None, max_dim will be used. 
            If both width and height are not None, max_dim will be ignored.
            Aspect ratio might not be preserved in this case.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        if self.max_dim is not None:
            self.check_bounds("max_dim", "[1, +inf)")
        if self.width is not None:
            self.check_bounds("width", "[1, +inf)")
        if self.height is not None:
            self.check_bounds("height", "[1, +inf)")            


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resizes an input image frame.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the key `img`.
        """
        orig_shape = inputs["img"].shape
        if self.height is None and self.width is None:
            max_dim = max(orig_shape)
            scale = self.max_dim / max_dim
            dsize = [
                int(round(orig_shape[1] * scale)), int(round(orig_shape[0] * scale))
            ]
        else:
            dsize = [self.width, self.height]
        img = cv2.resize(inputs["img"], dsize, interpolation=cv2.INTER_LINEAR)
        return {"img": img}


    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "max_dim": int,
            "width": Optional[int],
            "height": Optional[int],
        }
