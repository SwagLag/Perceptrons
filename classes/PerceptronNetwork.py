# The PerceptronNetwork houses **all** the layers of the network.

from classes.PerceptronLayer import PerceptronLayer

from typing import List, Union  # Onschuldige library die alleen beter laat zien wat voor soorten inputs er verwacht worden.

class PerceptronNetwork:
    """Defines the perceptron network; wraps all the given layers into this network."""
    def __init__(self, layers: List[PerceptronLayer]):
        """Initialises a perceptron network. Handles the connections between the layers."""
        self.hiddenlayers = layers
        self.input = []
        self.output = []

    def feed_forward(self, inputs: List[Union[int,float]]) -> List[Union[int,float]]:
        """Starts the network, feeds in the inputs, runs it through all the layers and returns the output
        of the final layer."""
        totalinputs = inputs.copy()  # Keep both lists unlinked; original list will be saved for debugging.
        for layer in self.hiddenlayers:
            layer.activate(totalinputs)
            totalinputs = layer.outputs.copy()  # Same deal here

        self.input = inputs
        self.output = totalinputs
        return totalinputs
