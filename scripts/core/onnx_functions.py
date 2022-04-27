import onnx
from onnx import utils, numpy_helper as onnx_nh

import numpy as np
"""Definition of ONNX Model functions"""

class OnnxFunctions():
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.model = onnx.shape_inference.infer_shapes(self.model) # Adds intermediate value info

    def get_model(self):
        return self.model
        
    def print(self):
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
        names = {"inputs": input_names, "outputs": output_names, "weights": weight_names}
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
        return next((weight for weight in self.model.graph.initializer if weight.name == name), None)

    def set_node_names(self):
        """Set node name if node name is empty. Required to perform operations such as combining two models"""
        for index, node in enumerate(self.model.graph.node):
            if not node.name:
                node.name = node.op_type + "_" + str(index)

    def set_weight(self, name=None, index=0, new_weight=None):
        if(name):
            weight = self.get_weight_by_name(name)
        else:
            try:
                weight = self.model.graph.initializer[index]
            except IndexError:
                print("Initializer index incorrect.")
                return
        weight_shape = tuple([i for i in weight.dims])
        new_shape = np.shape(new_weight)
        if(new_shape != weight_shape):
            print("New weight shape {} mismatch with current weight shape {}".format(new_shape, weight_shape))
            return

        weight.ClearField("float_data")
        weight.ClearField("int32_data")
        weight.ClearField("int64_data")
        weight.raw_data = new_weight.tobytes()
        return onnx_nh.to_array(weight)

    def check_model(self):
        """Validate ONNX graph"""
        onnx.checker.check_model(self.model)

    def save_model(self, filepath):
        """Save ONNX model to file"""
        onnx.save(self.model, filepath)
