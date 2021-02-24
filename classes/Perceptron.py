# Perceptron class. Takes a list of inputs, applies a list of weights to them, adds a bias,
# then applies the sum of that to the activation function and returns an output.

from typing import List, Union

class Perceptron:
    """Perceptron class. To initialise, takes a list of weights, an activation function and a bias (optional).
    Once initialised, can be activated by giving a list of inputs (with equal elements to the amount of weights)"""
    def __init__(self, weights: List[Union[int, float]], activation: callable, ID=0, bias: Union[int,float] = 0.0, learningrate: Union[int,float] = 0.1):
        """Initialises the perceptron."""
        # FUNCTIONAL VARIABLES (Private)
        self.__weights = weights
        self.__activation = activation
        self.__bias = bias
        self.__learningrate = learningrate

        # LOGGING VARIABLES (Public)
        self.ID = ID  # Identifier for Perceptron, for debugging.

        self.wastrained = False
        self.epochs = 0
        self.iterations = 0
        self.trainingerror = 0

        self.hasrun = False  # Whether the neuron has been activated or not.
        self.input = []  # Inputs of previous activations
        self.output = []  # Output of previous activations

    def getweights(self) -> List[Union[int, float]]:
        """Returns the current weights."""
        return self.__weights

    def setweights(self,weights: List[Union[int, float]]):
        """Changes the weights on this perceptron by using a supplied weightslist.
        For proper use in the PerceptronLayer class, the input has to have the same
        amount of elements as the original weights list."""
        if not len(weights) == len(self.getweights()):
            raise Exception("Amount of supplied weights does not equal the amount of current weights @ Perceptron {}".format(self.ID))
        self.__weights = weights

    def getlearningrate(self) -> Union[int,float]:
        return self.__learningrate

    def setlearningrate(self, learningrate: Union[int,float]):
        self.__learningrate = learningrate

    def getactivation(self) -> callable:
        """Returns the current activation function."""
        return self.__activation

    def setactivation(self, func: callable):
        """Changes the activation function on this perceptron."""
        self.__activation = func

    def getbias(self) -> Union[int, float]:
        """Returns the current bias for this perceptron."""
        return self.__bias

    def setbias(self, b: Union[int, float]):
        """Changes the current bias on this perceptron."""
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

    def update(self, inputs:List[List[int]], actualoutput: List[int], epochs: int = 40) -> None:
        """Calibrates the perceptron's weights based on a training set; it requires a list
        containing sets of inputs and a list containing the actual outputs. These are compared,
        and the perceptron learning rule is applied if necessary. An amount of maxiterations may
        be given as a failsafe to prevent a endless loop in case of a non-linear solvable problem."""
        self.wastrained = False
        self.epochs = 0
        self.iterations = 0
        sumerror = len(actualoutput)

        output = [0 for i in range(len(actualoutput))]
        weights = self.getweights()

        while sumerror > 0 and epochs > 0:
            for i in range(len(inputs)):  # Iterate over each input and expected output
                self.activate(inputs[i])  # Put the input in the neuron
                output[i] = self.output[-1]  # Extract the result
                error = actualoutput[i] - self.output[-1]  # Calculate the error
                weights = self.getweights()  # Get the current weights
                for j in range(len(weights)):  # Mutate each weight; weight = learningrate * error * input
                    weights[j] += (self.getlearningrate() * error * inputs[i][j])
                self.setbias(self.getbias() + (self.getlearningrate() * error))  # bias = learningrate * error
                self.setweights(weights)
                self.iterations += 1

            sumerror = self.error(output,actualoutput)
            epochs -= 1
            self.epochs += 1  # For logging purposes

        self.wastrained = True  # For logging purposes
        self.trainingerror = sumerror  # For logging purposes; MSE > 0 implies failed training

    def error(self, outputs:List[int], actualoutputs:List[int]) -> float:
        """Calculates the error of a perceptron's training set."""
        if not len(outputs) == len(actualoutputs):
            raise Exception("Either not enough testinputs or true outputs. @ Perceptron {}".format(self.ID))

        sumoutputs = []
        for indx in range(len(outputs)):
            sumoutputs.append((actualoutputs[indx] - outputs[indx])**2)

        return sum(sumoutputs) / len(outputs)

    def __str__(self) -> str:
        """Returns a string representing the object and it's variables."""
        output = ""
        output += "PERCEPTRON ID: {}\n\n".format(self.ID)

        output += "WEIGHTS: {}\n".format(self.getweights())
        output += "ACTIVATION: {}\n".format(self.getactivation().__name__)
        output += "BIAS: {}\n".format(self.getbias())

        if self.hasrun:
            output += "SUCCESFUL ACTIVATION \n\n".format(self.hasrun)
            output += "INPUT: {}\n".format(self.input)
            output += "OUTPUT: {}\n".format(self.output)
        else:
            output += "ACTIVATION PENDING/FAILED\n"

        if self.wastrained:
            output += "TRAINING STATISTICS: \n\n"
            output += "ITERATIONS: {}\n".format(self.iterations)
            output += "EPOCHS: {}\n".format(self.epochs)
            output += "MSE: {}\n".format(self.trainingerror)

        else:
            output += "NO TRAINING APPLIED\n"

        return output
