# Perceptron class. Takes a list of inputs, applies a list of weights to them, adds a bias,
# then applies the sum of that to the activation function and returns an output.

class Perceptron:
    """Perceptron class. To initialise, takes a list of weights, an activation function and a bias (optional).
    Once initialised, can be activated by giving a list of inputs (with equal elements to the amount of weights)"""
    def __init__(self,weights: list, activation: callable, ID=0, bias=0.0):
        """Initialises the perceptron."""
        # FUNCTIONAL VARIABLES
        self.weights = weights
        self.activation = activation
        self.bias = bias

        # LOGGING VARIABLES
        self.ID = ID  # Identifier for Perceptron, for debugging.
        self.hasrun = False  # Whether the neuron has been activated or not.
        self.input = []  # Inputs of the previous activation
        self.output = 0  # Output of the previous activation

    def activate(self,inputs: list):
        """Activates the Perceptron by supplying inputs."""
        # PRECHECKS
        if not len(inputs) == self.connections:
            raise Exception("Amount of inputs is not equal to the amount of weights @ Perceptron {}".format(self.ID))
        # PROCESSING
        weightedlist = []  # List with processed inputs (input*weight)

        for indx in range(len(self.weights)):
            weightedlist.append(self.weights[indx]*inputs[indx])

        output = self.activation(sum(weightedlist)+self.bias)
        # Consider evaluation succesful past this point; get logging variables.
        self.hasrun = True
        self.input = inputs
        self.output = output

        return output

    def __str__(self):
        """Returns a string representing the object and it's variables."""
        print("PERCEPTRON ID: {}\n\n".format(self.ID))

        print("WEIGHTS: {}\n".format(self.weights))
        print("ACTIVATION: {}\n".format(self.activation.__name__))
        print("BIAS: {}\n".format(self.bias))

        if self.hasrun:
            print("SUCCESFUL ACTIVATION \n\n".format(self.hasrun))
            print("INPUT: {}".format(self.input))
            print("OUTPUT: {}".format(self.output))
        else:
            print("ACTIVATION PENDING")