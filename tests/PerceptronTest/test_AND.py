import unittest

from classes.Activation import Step
from classes.Perceptron import Perceptron

class Perceptron_AND(unittest.TestCase):
    """Tests the Perceptron class by building a AND logic gate Perceptron"""

    def setUp(self):
        """Prepares a AND-type Perceptron"""
        self.AND_Perceptron = Perceptron([0.5,0.5],Step(1).activate,"AND")

    def test_AND_high(self):
        """Tests a scenario in which a Perceptron - designed to be an AND port - returns a high output (1)"""
        outcome = self.AND_Perceptron.activate([1,1])
        self.assertEqual(outcome, 1)

    def test_AND_low(self):
        """Tests some scenarios in which a Perceptron - designed to be an AND port - returns a low output (0)"""
        outcome = self.AND_Perceptron.activate([0,0])
        self.assertEqual(outcome, 0)

        outcome = self.AND_Perceptron.activate([0,1])
        self.assertEqual(outcome, 0)

        outcome = self.AND_Perceptron.activate([1,0])
        self.assertEqual(outcome, 0)