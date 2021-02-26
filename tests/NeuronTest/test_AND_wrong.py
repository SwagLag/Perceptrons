from classes.Activation import Sigmoid
from classes.Neuron import Neuron

AND_Neuron_wrong = Neuron([0.5, 0.5], Sigmoid().activate, bias= -1)

print("Should be high (1):")
outcome = AND_Neuron_wrong.activate([1, 1])
print("Input [1,1] gives {}".format(outcome))
print("")
print("Should be low (0):")
outcome = AND_Neuron_wrong.activate([1, 0])
print("Input [1,0] gives {}".format(outcome))
outcome = AND_Neuron_wrong.activate([0, 1])
print("Input [0,1] gives {}".format(outcome))
outcome = AND_Neuron_wrong.activate([0, 0])
print("Input [0,0] gives {}".format(outcome))