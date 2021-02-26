import unittest

from classes.Activation import Sigmoid
from classes.Neuron import Neuron

class Neuron_NOR(unittest.TestCase):
    """Tests the Neuron class by building a NOR logic gate Neuron"""
    def setUp(self):
        """Prepares a NOR-type Neuron"""
        self.NOR_Neuron = Neuron([-12,-12,-12],Sigmoid().activate,bias=6)

    def test_NOR_high(self):
        """Tests a scenario in which a Neuron - designed to be an NOR port - returns a output as close as possible to high (1)"""
        outcome = self.NOR_Neuron.activate([0,0,0])
        self.assertAlmostEqual(outcome, 1, 2)

    def test_NOR_low(self):
        """Tests some scenarios in which a Neuron - designed to be an NOR port - returns a output as close as possible to low (0)"""
        outcome = self.NOR_Neuron.activate([1, 0, 0])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.NOR_Neuron.activate([0, 1, 0])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.NOR_Neuron.activate([1, 1, 0])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.NOR_Neuron.activate([0, 0, 1])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.NOR_Neuron.activate([1, 0, 1])
        self.assertAlmostEqual(outcome, 0, 2)

        outcome = self.NOR_Neuron.activate([0, 1, 1])
        self.assertAlmostEqual(outcome, 0, 2)