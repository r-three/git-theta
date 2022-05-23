import onnx
from onnx import numpy_helper as nh
from onnx2pytorch import ConvertModel

import torch
"""Definition of ONNX Model functions"""


class AugumentedOnnxModel:
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.model = onnx.shape_inference.infer_shapes(
            self.model
        )  # Adds intermediate value info

    def get_onnx_model(self):
        return self.model

    def print_onnx_model(self):
        """Prints th ONNX model graph"""
        print(self.model)

    def get_input_output_weight_names(self):
        """Gets the names of inputs, outputs and intializers (weights)"""
        input_names = []
        output_names = []
        weight_names = []
        for input in self.model.graph.input:
            input_names.append(input.name)
        for output in self.model.graph.output:
            output_names.append(output.name)
        for weight in self.model.graph.initializer:
            weight_names.append(weight.name)
        names = {
            "inputs": input_names,
            "outputs": output_names,
            "weights": weight_names,
        }
        return names

    def change_names(self, name_map):
        """
        Changes the names of inputs, outputs and intializers (weights) in the graph and in individual nodes
        name_map: Takes a dictionary of keys=current_name, values=new_name
        Defaults to the existing name, if the key does not exist in the name_map
        """
        for input in self.model.graph.input:
            input.name = name_map.get(input.name, input.name)
        for output in self.model.graph.output:
            output.name = name_map.get(output.name, output.name)
        for weight in self.model.graph.initializer:
            weight.name = name_map.get(weight.name, weight.name)

        for node in self.model.graph.node:
            for index, input in enumerate(node.input):
                node.input[index] = name_map.get(input, input)
            for index, output in enumerate(node.output):
                node.output[index] = name_map.get(output, output)

    def get_weight_by_name(self, name):
        """Returns the intializer in the graph corresponding to the name, return None if non-existent"""
        return next(
            (weight for weight in self.model.graph.initializer if weight.name == name),
            None,
        )

    def update_initializer(self, initializer_name, update_type, delta, diff_order):
        initializer = self.get_weight_by_name(initializer_name)
        current_value = nh.to_array(initializer)

        if update_type == "dense":
            new_value = current_value + delta if diff_order == "next_diff" else current_value - delta
        elif update_type == "low_rank":
            low_rank_product = delta[0] @ delta[1]
            new_value = current_value + low_rank_product if diff_order == "next_diff" else current_value - delta
        else:
            raise NotImplementedError(
                "Only the dense and low-rank update types are currently supported"
            )
        initializer.raw_data = new_value.tobytes()

    def set_node_names(self):
        """Set node name if node name is empty. Required to perform operations such as combining two models"""
        for index, node in enumerate(self.model.graph.node):
            if not node.name:
                node.name = node.op_type + "_" + str(index)

    def check_model(self):
        """Validate ONNX graph"""
        onnx.checker.check_model(self.model)

    def save_model(self, filepath):
        """Save ONNX model to file"""
        onnx.save(self.model, filepath)

    def equal_to(self, other_model):
        curr_pytorch_model = ConvertModel(self.get_onnx_model(), experimental=True)
        other_pytorch_model = ConvertModel(other_model.get_onnx_model(), experimental=True)
        is_equal = True

        param_pairs = list(zip(curr_pytorch_model.named_parameters(), other_pytorch_model.named_parameters()))
        for param in param_pairs:
            if torch.allclose(param[0][1], param[1][1]):
                print(f"{param[0][0]} i.e. {param[1][0]} -: EQUAL")
            else:
                is_equal = False
                print(f"{param[0][0]} i.e. {param[1][0]} -: NOT EQUAL")

        return is_equal
