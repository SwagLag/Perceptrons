import unittest

from classes.Activation import Step
from classes.Perceptron import Perceptron

class Perceptron_INVERT(unittest.TestCase):
    """Tests the Perceptron class by building a INVERT logic gate Perceptron"""

    def setUp(self):
        """Prepares a INVERT-type Perceptron."""
        self.INVERT_Perceptron = Perceptron([-1],Step(0).activate,"INVERT")

    def test_INVERT_high(self):
        """Tests a scenario in which a Perceptron - designed to be an INVERTER - returns a high output (1)"""
        outcome = self.INVERT_Perceptron.activate([0])
        self.assertEqual(outcome, 1)

    def test_INVERT_low(self):
        """Tests a scenario in which a Perceptron - designed to be an INVERTER - returns a low output (0)"""
        outcome = self.INVERT_Perceptron.activate([1])
        self.assertEqual(outcome, 0)