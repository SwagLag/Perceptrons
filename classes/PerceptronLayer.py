# Perceptron layer class. Actually defines 3 types of layers that are necessary in the Perceptron Network class.

# InputLayer defines an input layer. It takes a list equal to the amount of nodes in the InputLayer to start
# evaluating the entire Perceptron Network.

# OutputLayer defines an output layer. The outputs registered here will be returned after the network has finished
# evaluating.

# PerceptronLayer defines the hidden layers in the network. This is where the Perceptron Class is used.

# By default, all the Layers are connected with eachother, however, all the weights amount to 0. You will need
# to manually define (for now) how much weight each connection should have (in the Perceptron node of each layer)

class InputLayer:
    """InputLayer defines an input layer. It takes a list equal to the amount of nodes in the InputLayer to start
    evaluating the entire Perceptron Network. Only one should ever exist in the PerceptronNetwork class."""

class HiddenLayer:
    """HiddenLayer defines the hidden layers in the network. This is where the perceptron classes are used.
    There can be as many as possible, but ultimately there should be atleast one."""

class OutputLayer:
    """OutputLayer defines an output layer. The outputs registered here will be returned after the network has finished
    evaluating. Only one should ever exist in the PerceptronNetwork class."""