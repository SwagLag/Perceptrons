import unittest

from classes.Activation import Step
from classes.Perceptron import Perceptron

class Perceptron_OR(unittest.TestCase):
    """Tests the Perceptron class by building a OR logic gate Perceptron"""
    def setUp(self):
        """Prepares a OR-type Perceptron"""
        self.OR_Perceptron = Perceptron([0.5, 0.5], Step(0.5).activate, "OR")

    def test_OR_high(self):
        """Tests some scenarios in which a Perceptron - designed to be an OR port - returns a high output (1)"""
        outcome = self.OR_Perceptron.activate([1, 1])
        self.assertEqual(outcome, 1)

        outcome = self.OR_Perceptron.activate([1, 0])
        self.assertEqual(outcome, 1)

        outcome = self.OR_Perceptron.activate([0, 1])
        self.assertEqual(outcome, 1)

    def test_OR_low(self):
        """Tests some scenarios in which a Perceptron - designed to be an OR port - returns a low output (0)"""
        outcome = self.OR_Perceptron.activate([0, 0])
        self.assertEqual(outcome, 0)