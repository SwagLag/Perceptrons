# The NeuronLayer houses **all** the layers of the network.

from classes.NeuronLayer import NeuronLayer

from typing import List, Union, Any  # Onschuldige library die alleen beter laat zien wat voor soorten inputs er verwacht worden.

class NeuronNetwork:
    """Defines the neuron network; wraps all the given layers into this network."""
    def __init__(self, layers: List[NeuronLayer], ID: Any = 0):
        """Initialises a neuron network. Handles the connections between the layers."""
        self.hiddenlayers = layers
        self.input = []
        self.output = []

        self.ID = ID
        self.hasrun = False

    def feed_forward(self, inputs: List[Union[int,float]]) -> List[Union[int,float]]:
        """Starts the network, feeds in the inputs, runs it through all the layers and returns the output
        of the final layer."""
        self.hasrun = False
        totalinputs = inputs.copy()  # Keep both lists unlinked; original list will be saved for debugging.
        for layer in self.hiddenlayers:
            layer.activate(totalinputs)
            totalinputs = layer.outputs.copy()  # Same deal here

        self.input = inputs
        self.output = totalinputs
        self.hasrun = True

        return totalinputs

    def __str__(self):
        """Tries to print out the network in a readable manner.
        Additional information is available once the network has been run once."""
        output = ""
        output += "NEURONNETWORK ID: {}\n".format(self.ID)
        if self.hasrun:
            output += "INPUT: {}\nV\n".format(self.input)
        for layer in self.hiddenlayers:
            output += "{}\n".format([i.getweights() for i in layer.neurons])
            if self.hasrun:
                output += "OUTPUT: {}\n".format(layer.outputs)
            output += "V\n"
        if self.hasrun:
            output += "FINAL OUTPUT: {}\n".format(self.output)
        else:
            output += "ACTIVATION PENDING\n"
        return output