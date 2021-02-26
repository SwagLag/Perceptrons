import unittest

from classes.Activation import Sigmoid

from classes.Neuron import Neuron
from classes.NeuronLayer import NeuronLayer
from classes.NeuronNetwork import NeuronNetwork

class NeuronNetwork_Halfadder(unittest.TestCase):
    """Builds and tests a neuron network based on the Half adder."""
    def setUp(self):
        """Prepares a Halfadder Neuron Network"""
        # Layer 1 Neurons:
        n1 = Neuron([12,12], Sigmoid().activate, bias=-18)
        n2 = Neuron([-12,-12], Sigmoid().activate, bias=6)
        n3 = Neuron([12,12], Sigmoid().activate, bias=-18)
        # Layer 2 Neurons:
        n4 = Neuron([-12,-12,0], Sigmoid().activate, bias=6)
        n5 = Neuron([0,0,12], Sigmoid().activate, bias=-6)
        # Layers
        l1 = NeuronLayer([n1,n2,n3])
        l2 = NeuronLayer([n4,n5])

        self.ntwrk = NeuronNetwork([l1,l2])

    def test_HA_sum(self):
        """Tests some scenarios in which the output of sum would be high (1), and the carry low (0)"""
        outcome = self.ntwrk.feed_forward([0,1])
        self.assertAlmostEqual(outcome[0], 1, 2)
        self.assertAlmostEqual(outcome[1], 0, 2)

        outcome = self.ntwrk.feed_forward([1,0])
        self.assertAlmostEqual(outcome[0], 1, 2)
        self.assertAlmostEqual(outcome[1], 0, 2)

    def test_HA_carry(self):
        """Tests a scenario in which the output of carry would be high (1), and sum low (0)"""
        outcome = self.ntwrk.feed_forward([1,1])
        self.assertAlmostEqual(outcome[0], 0, 2)
        self.assertAlmostEqual(outcome[1], 1, 2)

    def test_HA_low(self):
        """Tests a scenario in which both outputs would be low (0)"""
        outcome = self.ntwrk.feed_forward([0,0])
        self.assertAlmostEqual(outcome[0], 0, 2)
        self.assertAlmostEqual(outcome[1], 0, 2)
