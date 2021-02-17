import random
random.seed(1756708)

from classes.Perceptron import Perceptron
from classes.Activation import Step

test1 = Perceptron([random.random() for x in range(2)],Step(0).activate)

test1.update([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1],20)
print(test1.__str__())
