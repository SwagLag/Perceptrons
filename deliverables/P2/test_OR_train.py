from Perceptron import Perceptron
from Activation import Step

import random
random.seed(1756708)

test1 = Perceptron([random.random() for x in range(2)], Step(0).activate)
test1.update([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1])

# Uncomment voor uitgebreide informatie
# print(test1.__str__())

print("End weights:")
print(test1.getweights())
print("End bias:")
print(test1.getbias())

print("")

print("Should be high (1):")
print("Input [1,1] gives:")
print(test1.activate([1,1]))
print("")
print("Should be low (0):")
print("Input [0,0] gives:")
print(test1.activate([0,0]))
print("Input [0,1] gives:")
print(test1.activate([0,1]))
print("Input [1,0] gives:")
print(test1.activate([1,0]))



# class Perceptron_OR_traintest(unittest.TestCase):
#     """Attempts to train a perceptron using the inputs/outputs from OR to get
#     the OR gate's behaviour. Since an OR perceptron is a linear problem, the tests
#     will be geared around confirming that the behaviour is sufficient."""
#
#     def setUp(self):
#         self.trained_AND_Perceptron = Perceptron([random.random() for x in range(2)], Step(0).activate)
#         self.trained_AND_Perceptron.update([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1], 20)
#
#     def test_high(self):
#         """Tests the trained perceptron if it matches the logic of an OR port, in this case; whether
#         the high output is returned properly."""
#         outcome = self.trained_AND_Perceptron.activate([1, 1])
#         self.assertEqual(outcome, 1)
#
#     def test_low(self):
#         """Tests the trained perceptron if it matches the logic of an OR port, in this case; whether
#         the high output is returned properly."""
#         outcome = self.trained_AND_Perceptron.activate([0, 1])
#         self.assertEqual(outcome, 0)
#
#         outcome = self.trained_AND_Perceptron.activate([1, 0])
#         self.assertEqual(outcome, 0)
#
#         outcome = self.trained_AND_Perceptron.activate([0, 0])
#         self.assertEqual(outcome, 0)
