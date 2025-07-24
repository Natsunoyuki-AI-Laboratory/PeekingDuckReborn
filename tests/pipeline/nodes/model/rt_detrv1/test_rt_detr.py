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
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml
from typeguard import TypeCheckError

from peekingduck.pipeline.nodes.model.rt_detr import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def rt_detr_config():
    with open(PKD_DIR / "configs" / "model" / "rt_detr.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["model_path"] = "PekingU/rtdetr_r18vd"
    node_config["root"] = Path.cwd()
    # The following prevents pytest from creating `peekingduck_weights/`
    # in the parent directory of `PeekingDuckReborn/`.
    node_config["weights_parent_dir"] = str(PKD_DIR.parent)
    return node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def rt_detr_bad_config_value(request, rt_detr_config):
    rt_detr_config[request.param["key"]] = request.param["value"]
    return rt_detr_config


# TODO add the other RT-DETR models such as "rtdetr_r34vd", "rtdetr_r50vd".
@pytest.fixture(params=["PekingU/rtdetr_r18vd"])
def rt_detr_config_cpu(request, rt_detr_config):
    rt_detr_config["model_path"] = request.param
    # Mock CUDA is not available.
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield rt_detr_config


@pytest.mark.mlmodel
class TestRTDETR:
    def test_no_human_image(self, no_human_image, rt_detr_config_cpu):
        no_human_img = cv2.imread(no_human_image)
        node = Node(rt_detr_config_cpu)
        output = node.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
    

    def test_detect_human_bboxes(self, human_image, rt_detr_config_cpu):
        human_img = cv2.imread(human_image)
        node = Node(rt_detr_config_cpu)
        output = node.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = node.config["model_path"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_detect_human_bboxes_gpu(self, human_image, rt_detr_config):
        human_img = cv2.imread(human_image)
        node = Node(rt_detr_config)
        output = node.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = node.config["model_path"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)


    def test_get_detect_ids(self, rt_detr_config):
        node = Node(rt_detr_config)
        assert node.model.detect_ids == [0]


    def test_invalid_config_detect_ids(self, rt_detr_config):
        rt_detr_config["detect"] = 1
        with pytest.raises(TypeCheckError):
            _ = Node(config=rt_detr_config)


    def test_invalid_config_value(self, rt_detr_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=rt_detr_bad_config_value)
        assert "score_threshold must be between [0.0, 1.0]" in str(excinfo.value)


    def test_invalid_image(self, no_human_image, rt_detr_config):
        no_human_img = cv2.imread(no_human_image)
        node = Node(rt_detr_config)
        # Potentially passing in a file path or a tuple from image reader output.
        with pytest.raises(TypeError) as excinfo:
            _ = node.run({"img": Path.cwd()})
        assert "image must be a np.ndarray" == str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            _ = node.run({"img": ("image name", no_human_img)})
        assert "image must be a np.ndarray" == str(excinfo.value)
