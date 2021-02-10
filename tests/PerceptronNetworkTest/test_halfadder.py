import unittest

from classes.Activation import Step

from classes.Perceptron import Perceptron
from classes.PerceptronLayer import PerceptronLayer
from classes.PerceptronNetwork import PerceptronNetwork

class PerceptronNetwork_Halfadder(unittest.TestCase):
    """Builds and tests a perceptron network based on the Half adder."""
    # Layer 1 Perceptrons:

    def setUp(self):
        n1 = Perceptron([0.5,0.5],Step(1).activate,"n1")
        n2 = Perceptron([-0.1,-0.1],Step(0).activate,"n2")
        n3 = Perceptron([0.5,0.5],Step(1).activate,"n3")
        # Layer 2 Perceptrons:
        n4 = Perceptron([-1,-1,0],Step(-0.5).activate,"n4")
        n5 = Perceptron([0,0,1],Step(1).activate,"n5")

        l1 = PerceptronLayer(1,[n1,n2,n3])
        l2 = PerceptronLayer(2,[n4,n5])

        self.ntwrk = PerceptronNetwork([l1,l2])

    def test_HA_sum1(self):
        """Tests some scenario's in which the output of sum would be high (1), and the carry low (0)"""
        outcome = self.ntwrk.feed_forward([0,1])
        self.assertEqual(outcome, [1,0])

    def test_HA_sum2(self):
        outcome = self.ntwrk.feed_forward([1,0])
        self.assertEqual(outcome, [1,0])

    def test_HA_carry(self):
        """Tests a scenario in which the output of carry would be high (1), and sum low (0)"""
        outcome = self.ntwrk.feed_forward([1,1])
        self.assertEqual(outcome, [0,1])

    def test_HA_low(self):
        """Tests some scenarios in which both outputs would be low (0)"""
        outcome = self.ntwrk.feed_forward([0,0])
        self.assertEqual(outcome, [0,0])