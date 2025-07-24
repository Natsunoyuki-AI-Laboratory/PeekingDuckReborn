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

from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.vit_pose import Node
from tests.conftest import PKD_DIR, get_groundtruth

TOLERANCE = 1e-5
GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def vit_pose_config():
    with open(PKD_DIR / "configs" / "model" / "vit_pose.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["model_path"] = "usyd-community/vitpose-plus-small"
    node_config["root"] = Path.cwd()
    # The following prevents pytest from creating `peekingduck_weights/`
    # in the parent directory of `PeekingDuckReborn/`.
    node_config["weights_parent_dir"] = str(PKD_DIR.parent)
    return node_config


@pytest.fixture(
    params=[
        {"key": "keypoint_score_threshold", "value": -0.5},
        {"key": "keypoint_score_threshold", "value": 1.5},
    ],
)
def vit_pose_bad_config_value(request, vit_pose_config):
    vit_pose_config[request.param["key"]] = request.param["value"]
    return vit_pose_config

# TODO
@pytest.mark.mlmodel
class TestVitPose:
    def test_invalid_config_value(self, vit_pose_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=vit_pose_bad_config_value)
        assert "keypoint_score_threshold must be between [0.0, 1.0]" in str(excinfo.value)


    def test_no_human_image(self, no_human_image, vit_pose_config):
        """Tests VITPose on images with no humans present."""
        no_human_img = cv2.imread(no_human_image)
        node = Node(vit_pose_config)
        output = node.run({"img": no_human_img, "bboxes": np.empty((0, 4))})
        expected_output = {
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
        }

        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i], err_msg=f"unexpected output for {i}"
            )


    def test_single_human(self, single_person_image, vit_pose_config):
        single_human_img = cv2.imread(single_person_image)
        node = Node(vit_pose_config)
        output = node.run(
            {
                "img": single_human_img,
                "bboxes": np.array([[0.19026423, 0.08217245, 0.59008735, 0.9059642]]),
            }
        )

        model_path = node.config["model_path"]
        image_name = Path(single_person_image).stem
        expected = GT_RESULTS[model_path][image_name]

        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=1e-4)
        npt.assert_allclose(
            output["keypoint_conns"], expected["keypoint_conns"], atol=1e-4
        )
        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=1e-3
        )


    @pytest.mark.skip("WIP")
    def test_multi_person(self, multi_person_image, hrnet_config):
        multi_person_img = cv2.imread(multi_person_image)
        hrnet = Node(hrnet_config)

        model_path = hrnet.config["model_path"]
        image_name = Path(multi_person_image).stem
        expected = GT_RESULTS[model_path][image_name]
        output = hrnet.run(
            {"img": multi_person_img, "bboxes": np.array(expected["bboxes"])}
        )

        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)

        assert len(output["keypoint_conns"]) == len(expected["keypoint_conns"])
        # Detections can have different number of valid keypoint connections
        # and the keypoint connections result can be a ragged list lists.  When
        # converted to numpy array, the `keypoint_conns`` array will become
        # np.array([list(keypoint connections array), list(next keypoint
        # connections array), ...])
        # Thus, iterate through the detections
        for i, expected_keypoint_conns in enumerate(expected["keypoint_conns"]):
            npt.assert_allclose(
                output["keypoint_conns"][i],
                expected_keypoint_conns,
                atol=TOLERANCE,
            )

        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
        )
