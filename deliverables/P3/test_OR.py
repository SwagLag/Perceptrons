import unittest

from Activation import Sigmoid
from Neuron import Neuron

class Neuron_OR(unittest.TestCase):
    """Tests the Neuron class by building a OR logic gate Neuron"""

    def setUp(self):
        """Prepares a OR-type Neuron"""
        self.OR_Neuron = Neuron([12, 12], Sigmoid().activate, bias=-6)

    def test_OR_high(self):
        """Tests some scenarios in which a Neuron - designed to be an OR port - returns a output as close as possible to high (1)"""
        outcome = self.OR_Neuron.activate([1, 1])
        self.assertAlmostEqual(outcome, 1, 2)

        outcome = self.OR_Neuron.activate([1, 0])
        self.assertAlmostEqual(outcome, 1, 2)

        outcome = self.OR_Neuron.activate([0, 1])
        self.assertAlmostEqual(outcome, 1, 2)

    def test_OR_low(self):
        """Tests a scenario in which a Neuron - designed to be an OR port - returns a output as close as possible to low (0)"""
        outcome = self.OR_Neuron.activate([0, 0])
        self.assertAlmostEqual(outcome, 0, 2)
