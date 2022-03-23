import numpy as np
import pytest
from onnx import numpy_helper as onnx_nh
from onnx_functions import OnnxFunctions

class TestOnnxFunctions:
    @pytest.fixture
    def onnx_model(self):
        model = OnnxFunctions("onnx-models/mlp-model.onnx")
        return model

    def test_get_input_output_weight_names(self, onnx_model):
        names = {
            'inputs': ['input.1'],
            'outputs': ['12'],
            'weights': ['fc1.bias', 'fc2.bias', '13', '14']
        }

        assert names == onnx_model.get_input_output_weight_names()

    def test_change_names(self, onnx_model):
        name_map = {
            "12": "output",
            "13": "weights1",
            "14": "weights2",
        }

        new_names = {
            'inputs': ['input.1'],
            'outputs': ['output'],
            'weights': ['fc1.bias', 'fc2.bias', 'weights1', 'weights2']
        }

        onnx_model.change_names(name_map)
        assert new_names == onnx_model.get_input_output_weight_names()

    def test_get_weight_by_name(self, onnx_model):  
        expected_weights = np.array([0.0235439], dtype=np.float32)
        actual_weights = onnx_nh.to_array(onnx_model.get_weight_by_name('fc2.bias'))
        assert len(actual_weights) == len(expected_weights)
        assert all([a == b for a, b in zip(actual_weights, expected_weights)])

    def test_set_node_names(self, onnx_model):
        name = "MatMul_0"
        assert name == onnx_model.model.graph.node[0].name
        onnx_model.model.graph.node[0].name = ""
        assert "" == onnx_model.model.graph.node[0].name
        onnx_model.set_node_names()
        assert name == onnx_model.model.graph.node[0].name
