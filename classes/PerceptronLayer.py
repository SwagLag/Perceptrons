# Perceptron layer class. Actually defines 3 types of layers that are necessary in the Perceptron Network class.

# InputLayer defines an input layer. It takes a list equal to the amount of nodes in the InputLayer to start
# evaluating the entire Perceptron Network.

# OutputLayer defines an output layer. The outputs registered here will be returned after the network has finished
# evaluating.

# PerceptronLayer defines the hidden layers in the network. This is where the Perceptron Class is used.

# By default, all the Layers are connected with eachother, however, all the weights amount to 0. You will need
# to manually define (for now) how much weight each connection should have (in the Perceptron node of each layer)

from classes.Perceptron import Perceptron
from typing import List

class PerceptronLayer:

    def __init__(self, ID, perceptrons: List[Perceptron]):
        self.perceptrons = perceptrons
        self.outputs = []
        self.neuroncount = len(perceptrons)

    # def __init__(self, ID, weights: [List[List[int or float]]], bias: List[int or float], activation: List[callable]):
    #     self.perceptrons = []
    #     self.outputs = []
    #     self.neuroncount = 0
    #
    #     if not len(weights) == len(bias) or not len(bias) == len(activation) or not len(activation) == len(weights):
    #         raise Exception("Invalid arguments given @ PerceptronLayer {}".format(ID))
    #
    #     for i in range(len(weights)):
    #         self.perceptrons.append(Perceptron(weights[i],activation[i],i,bias[i]))
    #         self.neuroncount += 1

    def activate(self, inputlist: List[List[int or float]]):
        self.outputs = []
        for i in range(len(inputlist)):
            self.perceptrons[i].activate(inputlist[i])
            self.outputs.append(self.perceptrons[i].output)
# class InputLayer:
#     """InputLayer defines an input layer. It takes a list equal to the amount of nodes in the InputLayer to start
#     evaluating the entire Perceptron Network. Only one should ever exist in the PerceptronNetwork class."""
#     def __init__(self, ID, nodes: int):
#         """Initialises an input layer."""
#         self.inputs = [None for y in range(nodes)]
#         self.nodes = nodes
#
#
# class HiddenLayer:
#     """HiddenLayer defines the hidden layers in the network. This is where the perceptron classes are used.
#     There can be as many as possible, but ultimately there should be atleast one."""
#     def __init__(self, ID, inppn:int, weights: List[List[int or float]], bias: List[int or float], activation: List[callable]):
#         """initialises an hidden layer."""
#         if not len(weights) == len(bias) or not len(weights) == len(activation) or not len(bias) == len(activation):
#             raise Exception("Given argument lengths not equal @ PerceptronLayer {}".format(ID))
#         self.perceptrons = []
#         for indx1 in range(len(weights)):
#             self.perceptrons.append(Perceptron(weights[indx1],activation[indx1],indx1,bias[indx1]))
#
#         self.weights = weights
#         self.bias = bias
#         self.activation = activation
#
#         self.ID = ID
#
#     def check_connections(self):
#         """Checks if sufficient connections have been realised."""
#
# class OutputLayer:
#     """OutputLayer defines an output layer. The OutputLayer works like a HiddenLayer in essence, but signalifies the
#      end of the network; as a result, this Layer has weights unlike InputLayer. Only one should ever exist in the
#      PerceptronNetwork class."""
#
#     def __init__(self, ID, inppn: int, weights: List[List[int or float]], bias: List[int or float],
#                  activation: List[callable]):
#         """initialises an hidden layer."""
#         if not len(weights) == len(bias) or not len(weights) == len(activation) or not len(bias) == len(activation):
#             raise Exception("Given argument lengths not equal @ PerceptronLayer {}".format(ID))
#         self.inputs = [[] for x in range(inppn)]
#         self.nodes =