import random
import unittest
random.seed(1756708)

from classes.Perceptron import Perceptron
from classes.Activation import Step

class Perceptron_OR_traintest(unittest.TestCase):
    """Attempts to train a perceptron using the inputs/outputs from OR to get
    the OR gate's behaviour. Since an OR perceptron is a linear problem, the tests
    will be geared around confirming that the behaviour is sufficient."""

    def setUp(self):
        self.test1 = Perceptron([random.random() for x in range(2)], Step(0).activate)