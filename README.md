# collaborative-learning
Repo for the Collaborative Learning Project

## ONNX Model Architectures
mlp-model.onnx
- MLP with 5 inputs, 1 output, hidden layer with 3 nodes
- RELU applied to hidden layer output, sigmoid applied to output layer

## Steps to Run Checkout
- First setup folders by calling DiffTools.setup. The base onnx model should be stored under "models/"
- Run pytorch_cifar_mnist.ipynb to create a few diff files and the model head (model tail is same as the original onnx model)
- forward_model = diffops.checkout('181395534a90c3c357a45671ee209850.modeldiff', direction="forward")	# provide the name of one of the diff files
- backward_model = diffops.checkout('181395534a90c3c357a45671ee209850.modeldiff', direction="backward") # get same checkpoint model through backward traversal
- forward_model.equal_to(backward_model) # To check if all the parameter values are equal
- if direction == "auto" in checkout function, it uses whichever traversal requires lesser number of diffs