import unittest

from classes.Activation import Sigmoid
from classes.Neuron import Neuron

class Neuron_INVERT(unittest.TestCase):
    """Tests the Neuron class by building a INVERT logic gate Neuron"""

    def setUp(self):
        """Prepares a INVERT-type Neuron."""
        self.INVERT_Neuron = Neuron([-12], Sigmoid().activate, bias=6)

    def test_INVERT_high(self):
        """Tests a scenario in which a Neuron - designed to be an INVERT port - returns a output as close as possible to high (1)"""
        outcome = self.INVERT_Neuron.activate([0])
        self.assertAlmostEqual(outcome, 1, 2)

    def test_INVERT_low(self):
        """Tests a scenario in which a Neuron - designed to be an INVERT port - returns a output as close as possible to low (0)"""
        outcome = self.INVERT_Neuron.activate([1])
        self.assertAlmostEqual(outcome, 0, 2)
