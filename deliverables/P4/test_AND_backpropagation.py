from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork
from Activation import Sigmoid

n1 = Neuron([-0.5,0.5],Sigmoid().activate,bias=1.5)

l1 = NeuronLayer([n1])

ntwrk = NeuronNetwork([l1],1)

x = [[0,0],[0,1],[1,0],[1,1]]
y = [[0],[0],[0],[1]]

ntwrk.train(x,y,1000,0.0000001)
print(ntwrk.__str__())

print("MSE network:")
print(ntwrk.error(x,y))

print("Should be as close as possible to high (1)")
print("[1,1] gives:")
print(ntwrk.feed_forward([1,1]))
print("Should be as close as possible to low (0)")
print("[0,1] gives:")
print(ntwrk.feed_forward([0,1]))
print("[1,0] gives:")
print(ntwrk.feed_forward([1,0]))
print("[0,0] gives:")
print(ntwrk.feed_forward([0,0]))

