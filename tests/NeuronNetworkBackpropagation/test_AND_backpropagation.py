from classes.Neuron import Neuron
from classes.NeuronLayer import NeuronLayer
from classes.NeuronNetwork import NeuronNetwork
from classes.Activation import Sigmoid

n1 = Neuron([-0.5,0.5],Sigmoid().activate,bias=1.5)

l1 = NeuronLayer([n1])

ntwrk = NeuronNetwork([l1],1)

ntwrk.train([[0,0],[0,1],[1,0],[1,1]],[[0],[0],[0],[1]],1000,0.0000001)
print(ntwrk.__str__())
