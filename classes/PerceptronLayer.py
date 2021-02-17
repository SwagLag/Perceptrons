# PerceptronLayer defines the layers in the network. This is where the Perceptron Class is used.

# By default, all the Layers must be connected, so each perceptron in the network must have equally as many
# weights as one another, and the amount of weights on all perceptrons should equal the amount of connections from one layer to the
# other on a per-perceptron basis.

from classes.Perceptron import Perceptron
from typing import List, Union

class PerceptronLayer:
    """Defines a layer in a PerceptronNetwork."""
    def __init__(self, ID, perceptrons: List[Perceptron]):
        self.perceptrons = perceptrons
        self.outputs = []

    def activate(self, inputlist: List[Union[int, float]]):
        """Runs the inputlist through all perceptrons of the network and saves the output."""
        self.outputs = []
        for i in self.perceptrons:
            i.activate(inputlist)
            self.outputs.append(i.output[-1])