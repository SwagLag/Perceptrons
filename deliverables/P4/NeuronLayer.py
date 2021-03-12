# NeuronLayer defines the layers in the network. This is where the Neuron Class is used.

# By default, all the Layers must be connected, so each neuron in the network must have equally as many
# weights as one another, and the amount of weights on all perceptrons should equal the amount of connections from one layer to the
# other on a per-neuron basis.

from Neuron import Neuron
from typing import List, Union, Any

class NeuronLayer:
    """Defines a layer in a NeuronNetwork."""
    def __init__(self, neurons: List[Neuron], ID: Any = 0):
        self.neurons = neurons
        self.outputs = []

    def activate(self, inputlist: List[Union[int, float]]):
        """Runs the inputlist through all perceptrons of the network and saves the output."""
        self.outputs = []
        for i in self.neurons:
            i.activate(inputlist)
            self.outputs.append(i.output[-1])