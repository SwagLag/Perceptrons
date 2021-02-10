import unittest

from classes.Activation import Step

from classes.Perceptron import Perceptron
from classes.PerceptronLayer import PerceptronLayer
from classes.PerceptronNetwork import PerceptronNetwork

class PerceptronNetwork_XOR(unittest.TestCase):
    """Builds and tests a perceptron network based on the XOR logic port."""

    def setUp(self):
        # Layer 1 Perceptrons:
        n1 = Perceptron([0.5, 0.5], Step(1).activate, "n1")
        n2 = Perceptron([-0.1, -0.1], Step(0).activate, "n2")
        # Layer 2 Perceptrons:
        n3 = Perceptron([-1, -1], Step(-0.5).activate, "n3")

        l1 = PerceptronLayer(1, [n1,n2])
        l2 = PerceptronLayer(1, [n3])

        self.ntwrk = PerceptronNetwork([l1,l2])

    def test_XOR_high1(self):
        """Tests some scenario's in which the output would be high (1)"""
        outcome = self.ntwrk.feed_forward([0,1])
        self.assertEqual(outcome, [1])

    def test_XOR_high2(self):
        outcome = self.ntwrk.feed_forward([1,0])
        self.assertEqual(outcome, [1])

    def test_XOR_low1(self):
        """Tests some scenarios in which the output would be low (0)"""
        outcome = self.ntwrk.feed_forward([0,0])
        self.assertEqual(outcome, [0])

    def test_XOR_low2(self):
        outcome = self.ntwrk.feed_forward([1,1])
        self.assertEqual(outcome, [0])
