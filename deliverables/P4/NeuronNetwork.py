# The NeuronLayer houses **all** the layers of the network.

from NeuronLayer import NeuronLayer

from typing import List, Union, Any  # Onschuldige library die alleen beter laat zien wat voor soorten inputs er verwacht worden.

class NeuronNetwork:
    """Defines the neuron network; wraps all the given layers into this network."""
    def __init__(self, layers: List[NeuronLayer], learningrate: Union[int,float] = 0.3, ID: Any = 0,):
        """Initialises a neuron network. Handles the connections between the layers."""
        self.hiddenlayers = layers
        self.learningrate = learningrate
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

    def backpropagation(self, actualoutput):
        outputlayer = self.hiddenlayers[-1]

        if len(outputlayer.neurons) != len(actualoutput):
            raise Exception("Not enough outputs for each neuron in the output layer @ NeuronNetwork {}".format(self.ID))

        for i in range(len(outputlayer.neurons)):
            outputlayer.neurons[i].erroroutput(actualoutput[i],self.learningrate)
        # Nu komt het lastige gedeelte...
        # Amount of connections is equal to the amount of neurons in the previous layer!!
        # Currentlayer (i) : Target to call .errorhidden() on. Also get index in the neuron list.
        # Nextlayer (i+1) : Target to get weights from. Use index acquired in the previous layer.
        for lindx in range(len(self.hiddenlayers)-2,-1,-1):
            for i in range(len(self.hiddenlayers[lindx].neurons)):
                weights = []
                errors = []
                for neuron in self.hiddenlayers[lindx+1].neurons:
                    weights.append(neuron.getweights()[i])  # Gets the weights that this neuron connects to on neurons in the next layer
                    errors.append(neuron.error)  # Gets the error at the same time.
                self.hiddenlayers[lindx].neurons[i].errorhidden(weights,errors,self.learningrate)

    def update(self):
        """Updates all the weights and biases in the network immediately, given
        that all neurons have had their """
        for layer in self.hiddenlayers:
            for neuron in layer.neurons:
                neuron.update()

    def train(self, inputs: List[List[int]], actualoutputs: List[List[int]], epochs: int = 40, errortreshold: float = 0.1) -> None:
        error = errortreshold+1
        while epochs > 0 and error >= errortreshold:
            for i in range(len(inputs)):
                self.feed_forward(inputs[i])
                self.backpropagation(actualoutputs[i])
                self.update()
            error = self.error(inputs,actualoutputs)  # MSE
            epochs -= 1

    def error(self, inputs: List[List[Union[int,float]]], actualoutputs: List[List[Union[int,float]]]) -> float:
        """Calculates the MSE of this network's output layer over a training set."""
        outputs = []
        sumoutputs = []
        for i in range(len(inputs)):
            self.feed_forward(inputs[i])
            outputs.append(self.output)
            # Verwijder ook hier weer de resulterende inputs en outputs, die willen we niet; error moet gezien
            # worden als een functie zonder side-effects.
            for neuronlayer in self.hiddenlayers:
                for neuron in neuronlayer.neurons:
                    del neuron.output[-1]
                    del neuron.input[-1]

        for i1 in range(len(outputs)):
            for i2 in range(len(outputs[i1])):
                sumoutputs.append((actualoutputs[i1][i2] - outputs[i1][i2])**2)

        return sum(sumoutputs) / len(outputs)

    def __str__(self):
        """Tries to print out the network in a readable manner.
        Additional information is available once the network has been run once."""
        output = ""
        output += "NEURONNETWORK ID: {}\n".format(self.ID)
        if self.hasrun:
            output += "INPUT: {}\nV\n".format(self.input)
        for layer in self.hiddenlayers:
            for i in layer.neurons:
                output += "[{} + {}]\n".format([round(x,4) for x in i.getweights()],round(i.getbias(),4))
            if self.hasrun:
                output += "OUTPUT: {}\n".format(layer.outputs)
            output += "V\n"
        if self.hasrun:
            output += "FINAL OUTPUT: {}\n".format(self.output)
        else:
            output += "ACTIVATION PENDING\n"
        return output
