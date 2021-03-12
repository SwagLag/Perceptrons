from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork
from Activation import Sigmoid

# Layer 1
n1 = Neuron([0.2,-0.4],Sigmoid().activate)
n2 = Neuron([0.7,0.1],Sigmoid().activate)
# Layer 2
n3 = Neuron([0.6,0.9],Sigmoid().activate)

l1 = NeuronLayer([n1,n2])
l2 = NeuronLayer([n3])

ntwrk = NeuronNetwork([l1,l2],1)

x = [[0,0],[0,1],[1,0],[1,1]]
y = [[0],[1],[1],[0]]

ntwrk.train(x,y,40000,0.001)
print(ntwrk.__str__())

print("MSE network:")
print(ntwrk.error(x,y))

print("Should be as close as possible to high (1)")
print("[0,1] gives:")
print(ntwrk.feed_forward([0,1]))
print("[1,0] gives:")
print(ntwrk.feed_forward([1,0]))
print("Should be as close as possible to low (0)")
print("[1,1] gives:")
print(ntwrk.feed_forward([1,1]))
print("[0,0] gives:")
print(ntwrk.feed_forward([0,0]))