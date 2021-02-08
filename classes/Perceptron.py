# Perceptron class. Takes a list of inputs, applies a list of weights to them, adds a bias,
# then applies the sum of that to the activation function and returns an output.

class Perceptron:
    """Perceptron class. To initialise, takes a list of weights, an activation function and a bias (optional).
    Once initialised, can be activated by giving a list of inputs (with equal elements to the amount of weights)"""
    def __init__(self,weights: list, activation: callable, ID=0, bias=0.0):
        """Initialises the perceptron."""
        # FUNCTIONAL VARIABLES (Private)
        self.__weights = weights
        self.__activation = activation
        self.__bias = bias

        # LOGGING VARIABLES (Public)
        self.ID = ID  # Identifier for Perceptron, for debugging.
        self.hasrun = False  # Whether the neuron has been activated or not.
        self.input = []  # Inputs of the previous activation
        self.output = 0  # Output of the previous activation

    def getweights(self) -> list:
        """Returns the current weights."""
        return self.__weights

    def setweights(self,weights: list):
        """Changes the weights on this perceptron by using a supplied weightslist.
        For proper use in the PerceptronLayer class, the input has to have the same
        amount of elements as the original weights list."""
        if not len(weights) == len(self.getweights()):
            raise Exception("Amount of supplied weights does not equal the amount of current weights @ Perceptron {}".format(self.ID))
        self.__weights = weights

    def getactivation(self) -> callable:
        """Returns the current activation function."""
        return self.__activation

    def setactivation(self, func: callable):
        """Changes the activation function on this perceptron."""
        self.__activation = func

    def getbias(self) -> int or float:
        """Returns the current bias for this perceptron."""
        return self.__bias

    def setbias(self, b: int or float):
        """Changes the current bias on this perceptron."""
        self.__bias = b

    def activate(self,inputs: list):
        """Activates the Perceptron by supplying inputs."""
        # PRECHECKS
        if not len(inputs) == len(self.__weights):
            raise Exception("Amount of inputs is not equal to the amount of weights @ Perceptron {}".format(self.ID))
        # PROCESSING
        weightedlist = []  # List with processed inputs (input*weight)

        for indx in range(len(self.__weights)):
            weightedlist.append(self.__weights[indx] * inputs[indx])

        output = self.__activation(sum(weightedlist) + self.__bias)
        # Consider evaluation succesful past this point; get logging variables.
        self.hasrun = True
        self.input = inputs
        self.output = output

        return output

    def __str__(self):
        """Returns a string representing the object and it's variables."""
        print("PERCEPTRON ID: {}\n\n".format(self.ID))

        print("WEIGHTS: {}\n".format(self.getweights()))
        print("ACTIVATION: {}\n".format(self.getactivation().__name__))
        print("BIAS: {}\n".format(self.getbias()))

        if self.hasrun:
            print("SUCCESFUL ACTIVATION \n\n".format(self.hasrun))
            print("INPUT: {}".format(self.input))
            print("OUTPUT: {}".format(self.output))
        else:
            print("ACTIVATION PENDING/FAILED")