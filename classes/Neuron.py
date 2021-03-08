# Perceptron class. Takes a list of inputs, applies a list of weights to them, adds a bias,
# then applies the sum of that to the activation function and returns an output.

from typing import List, Union

class Neuron:
    """Neuron class. To initialise, takes a list of weights, an activation function (normally Sigmoid().activate()), and a bias (optional).
    Once initialised, can be activated by giving a list of inputs (with equal elements to the amount of weights)"""
    def __init__(self, weights: List[Union[int, float]], activation: callable, ID=0, bias: Union[int,float] = 0.0):
        """Initialises the neuron."""
        # FUNCTIONAL VARIABLES (Private)
        self.__weights = weights
        self.__activation = activation
        self.__bias = bias

        self.__newweights = []
        self.__newbias = 0

        # LOGGING VARIABLES (Public)
        self.ID = ID  # Identifier for Perceptron, for debugging.

        self.error = 0  # Error calculated by the error methods.

        self.hasrun = False  # Whether the neuron has been activated or not.
        self.input = []  # Inputs of previous activations
        self.output = []  # Output of previous activations

    def getweights(self) -> List[Union[int, float]]:
        """Returns the current weights."""
        return self.__weights

    def setweights(self,weights: List[Union[int, float]]):
        """Changes the weights on this neuron by using a supplied weightslist.
        For proper use in the PerceptronLayer class, the input has to have the same
        amount of elements as the original weights list."""
        if not len(weights) == len(self.getweights()):
            raise Exception("Amount of supplied weights does not equal the amount of current weights @ Perceptron {}".format(self.ID))
        self.__weights = weights

    def getactivation(self) -> callable:
        """Returns the current activation function."""
        return self.__activation

    def setactivation(self, func: callable):
        """Changes the activation function on this neuron."""
        self.__activation = func

    def getbias(self) -> Union[int, float]:
        """Returns the current bias for this neuron."""
        return self.__bias

    def setbias(self, b: Union[int, float]):
        """Changes the current bias on this neuron."""
        self.__bias = b

    def activate(self,inputs: List[Union[int, float]]) -> Union[int,float]:
        """Activates the Perceptron by supplying inputs."""
        # RESETS
        self.hasrun = False
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
        self.input.append(inputs)
        self.output.append(output)

        return output

    def erroroutput(self, target: Union[int,float], learningrate: Union[int,float]):
        """Calculates the error of an output neuron."""
        if not self.hasrun:
            raise Exception("Run the Neuron first! @ Neuron {}".format(self.ID))
        gradients = []
        deltaweights = []
        deltabias = 0
        newweights = []
        newbias = 0
        # Bepaal de error
        output = self.output[-1]
        error = output * (1-output) * -(target-output)
        for inp in self.input[-1]:
            gradients.append(inp * error)  # De output van een voorgaande node is gelijk aan de input op deze node op de relevante index
        for grad in gradients:
            deltaweights.append(learningrate * grad)
        deltabias = learningrate * error

        self.error = error
        self.__newweights = [self.__weights[i] - deltaweights[i] for i in range(len(self.getweights()))]
        self.__newbias = self.getbias() - deltabias

    def errorhidden(self, connections: List[Union[int,float]], errors: List[Union[int,float]], learningrate: Union[int,float]):
        """Calculates the error of a hidden layer neuron"""
        if not self.hasrun:
            raise Exception("Run the Neuron first! @ Neuron {}".format(self.ID))
        if len(connections) != len(errors):
            raise Exception("Amount of connections from this neuron should equal the amount of errors from neurons @ Neuron {}".format(self.ID))
        gradients = []
        deltaweights = []
        deltabias = 0
        newweights = []
        newbias = 0
        sum = 0  # Sum of (Wi,j * Delta(j))
        # Bepaal de error
        output = self.output[-1]
        for i in range(len(connections)):  # Bepaal eerst de som van de vermenigvuldigingen tussen de verbindingen en de errors.
            sum += connections[i] * errors[i]
        error = output * (1-output) * sum  # Bepaal dan uiteindelijk de error.
        for inp in self.input[-1]:
            gradients.append(inp * error)  # De output van een voorgaande node is gelijk aan de input op deze node op de relevante index
        for grad in gradients:
            deltaweights.append(learningrate * grad)
        deltabias = learningrate * error

        self.error = error
        self.__newweights = [self.__weights[i] - deltaweights[i] for i in range(len(self.getweights()))]
        self.__newbias = self.getbias() - deltabias

    def update(self):
        """Updates the weights and bias using stored new weights and bias."""
        self.setbias(self.__newbias)
        self.setweights(self.__newweights)

    def __str__(self) -> str:
        """Returns a string representing the object and it's variables."""
        output = ""
        output += "NEURON ID: {}\n\n".format(self.ID)

        output += "WEIGHTS: {}\n".format(self.getweights())
        output += "ACTIVATION: {}\n".format(self.getactivation().__name__)
        output += "BIAS: {}\n".format(self.getbias())

        if self.hasrun:
            output += "SUCCESFUL ACTIVATION \n\n".format(self.hasrun)
            output += "INPUT: {}\n".format(self.input)
            output += "OUTPUT: {}\n".format(self.output)
        else:
            output += "ACTIVATION PENDING/FAILED\n"

        return output
