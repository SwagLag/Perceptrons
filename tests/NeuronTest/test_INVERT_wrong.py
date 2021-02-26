from classes.Activation import Sigmoid
from classes.Neuron import Neuron

INVERTER_Neuron_wrong = Neuron([-1],Sigmoid().activate)

print("Should be high (1):")
outcome = INVERTER_Neuron_wrong.activate([0])
print("Input [0] gives {}".format(outcome))
print("")
print("Should be low (0):")
outcome = INVERTER_Neuron_wrong.activate([1])
print("Input [1] gives {}".format(outcome))