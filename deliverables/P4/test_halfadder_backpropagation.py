from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork
from Activation import Sigmoid

n1 = Neuron([0.0,0.1],Sigmoid().activate)
n2 = Neuron([0.2,0.3],Sigmoid().activate)
n3 = Neuron([0.4,0.5],Sigmoid().activate)

n4 = Neuron([0.6,0.7,0.8],Sigmoid().activate)
n5 = Neuron([0.9,1.0,1.1],Sigmoid().activate)

l1 = NeuronLayer([n1,n2,n3])
l2 = NeuronLayer([n4,n5])

ntwrk = NeuronNetwork([l1,l2],0.5)

x = [[0,0],[1,0],[0,1],[1,1]]
y = [[0,0],[1,0],[1,0],[0,1]]

ntwrk.train(x,y,80000,0.001)
print(ntwrk.__str__())

print("MSE network:")
print(ntwrk.error(x,y))

print("Output should be close to [0,0]")
print(ntwrk.feed_forward([0,0]))
print("Output should be close to [1,0]")
print(ntwrk.feed_forward([1,0]))
print(ntwrk.feed_forward([0,1]))
print("Output should be close to [0,1]")
print(ntwrk.feed_forward([1,1]))