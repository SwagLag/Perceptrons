# The PerceptronNetwork houses **all** the layers of the network.
# This wrapper class has a method that allows you to input values and get an output based on how
# the network (or the layers) is/are coded.

from classes.PerceptronLayer import PerceptronLayer

from typing import List  # Onschuldige library die alleen beter laat zien wat voor soorten inputs er verwacht worden.

def listmultiplier(target: list, amount: int):
    """Copies the list into a nested list, n amount of times."""
    return [target for n in range(amount)]

class PerceptronNetwork:
    """Defines the perceptron network; wraps all the given layers into this network."""
    def __init__(self, layers: List[PerceptronLayer]):
        """Initialises a perceptron network. Handles the connections between the layers."""
        self.hiddenlayers = layers

        self.input = []
        self.output = []

        self.hasrun = False

    def feed_forward(self, inputs: List[int or float]):
        """Starts the network, and returns the output."""
        if self.hasrun:  # Temporary workaround for weird bug. THIS NEEDS TO BE FIXED BEFORE A LEARNING ALGORITHM
                         # CAN BE APPLIED.
            raise Exception("Perceptron Network objects cannot be reused.")
        totalinputs = inputs.copy()
        for layer in self.hiddenlayers:
            temp = totalinputs.copy()  # For debugging.
            totalinputs = listmultiplier(totalinputs, layer.neuroncount)
            layer.activate(totalinputs)
            totalinputs = layer.outputs.copy()

        self.input = inputs
        self.output = totalinputs
        self.hasrun = True

        return totalinputs
