# Since the XOR gate solves a non-linear problem, we estimate that this perceptron will fail to initialise properly.
# In order to demonstrate this, we do **not** use the unittest module, as that would result in a lack of feedback
# as to why it went wrong. Instead, we will print the outcomes and show that the XOR port is not working as it should.

from classes.Perceptron import Perceptron
from classes.Activation import Step

import random
random.seed(1756708)

test1 = Perceptron([random.random() for x in range(2)], Step(0).activate)
test1.update([[0,0],[0,1],[1,0],[1,1]],[0,1,1,0])

# Uncomment voor uitgebreide informatie
# print(test1.__str__())

print("End weights:")
print(test1.getweights())
print("End bias:")
print(test1.getbias())

print("Should be high (1):")
print("Input [1,0] gives:")
print(test1.activate([1,0]))
print("Input [0,1] gives:")
print(test1.activate([0,1]))
print("")
print("Should be low (0):")
print("Input [0,0] gives:")
print(test1.activate([0,0]))
print("Input [1,1] gives:")
print(test1.activate([1,1]))

