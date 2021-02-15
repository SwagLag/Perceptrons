# Activation classes. The idea is as follows;
# The classes should have attributes, but should ultimately be callable to be used in the Perceptrons, in order
# to return an output.

# To this end, make sure that implemented classes have an activate() function that only takes a int or float input
# and outputs a int or float.

from typing import Union
import math

class Step:
    """Step-based activation. If the sum of the input is above the treshold, the output is 1. Otherwise,
    the output is 0."""
    def __init__(self, treshold: Union[int, float]):
        self.treshold = treshold

    def activate(self, input: Union[int, float]):
        if input >= self.treshold:
            return 1
        else:
            return 0

class Sigmoid:
    """Sigmoid-based activation. The output is defined by the sigmoid function."""
    def __init__(self):
        """Creates the object."""

    def activate(self, input: Union[int, float]):
        return 1 / (1 + math.e ** input)

