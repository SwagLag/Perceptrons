# Activation classes. The idea is as follows;
# The classes should have attributes, but should ultimately be callable to be used in the Perceptrons, in order
# to return an output.

# To this end, make sure that implemented classes have an activate() function that only takes a int or float input
# and outputs a int or float.

class Step:
    """Step-based activation. If the sum of the input is above the treshold, the output is 1. Otherwise,
    the output is 0."""
    def __init__(self, treshold: int or float):
        self.treshold = treshold

    def activate(self, input: int or float):
        if input >= self.treshold:
            return 1
        else:
            return 0
