import unittest

from Activation import Sigmoid
from Neuron import Neuron

class Neuron_AND(unittest.TestCase):
    """Tests the Neuron class by building a AND logic gate Neuron"""

    def setUp(self):
        """Prepares a AND-type Neuron"""
        self.AND_Neuron = Neuron([12,12], Sigmoid().activate, bias=-18)

    def test_AND_high(self):
        """Tests a scenario in which a Neuron - designed to be an AND port - returns a output as close as possible to high (1)"""
        outcome = self.AND_Neuron.activate([1,1])
        self.assertAlmostEqual(outcome, 1, 2)

    def test_AND_low(self):
        """Tests some scenarios in which a Neuron - designed to be an AND port - returns a output as close as possible to low (0)"""
        outcome = self.AND_Neuron.activate([0, 0])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.AND_Neuron.activate([0, 1])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.AND_Neuron.activate([1, 0])
        self.assertAlmostEqual(outcome, 0, 2)