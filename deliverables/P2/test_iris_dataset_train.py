from sklearn.datasets import load_iris
from typing import List, Union
import random
random.seed(1756708)

from Perceptron import Perceptron
from Activation import Step

data = load_iris()

x, y = data['data'], data['target']

# Uncomment om de geïmporteerde data te zien.
# print(x)
# print(y)

# Beetje data preparation; we willen eerst alleen de classes en bijbehorende gegevens voor Setosa en
# Versicolour, waarbij we er (hopelijk!!!!) er vanuit gaan dat beide soorten respectievelijk klasses 0 en 1 zijn.

def listfilter(wantedclasses: List[int], x: List[Union[int, float]], y: List[int]):
    """Gets the trainingset that only involves the wanted classes."""
    finalx = []
    finaly = []
    for index in range(len(y)):
        if y[index] in wantedclasses:
            finalx.append(list(x[index]))
            finaly.append(y[index])
    return finalx, finaly

firstx, firsty = listfilter([0,1],x,y)

# Train dan het eerste perceptron! We hebben hier te maken met twee trainingssets,
# dus het probleem zou wel lineair op te lossen moeten zijn.

percep = Perceptron([random.random() for i in range(4)], Step().activate)
percep.update(firstx,firsty)

# Uncomment voor uitgebreide informatie
# print(percep.__str__())

print("End weights:")
print(percep.getweights())
print("End bias:")
print(percep.getbias())
print("")

# Train dan nu het tweede perceptron! Deze zou iets lastiger kunnen zijn, omdat we hier te maken
# hebben met drie klasses; we werken immers met Step() die alleen maar 0 of 1 kan geven.... Bovendien
# werken we in deze opdracht alleen maar met een enkele perceptron die getraind wordt. Als we een netwerk
# zouden maken zou dit geen probleem geweest zijn; dan hadden we voor elke klasse een node in de output-layer
# kunnen maken.

percep2 = Perceptron([random.random() for j in range(4)], Step().activate)
percep.update(x,y)

# Uncomment voor uitgebreide informatie
# print(percep.__str__())

print("End weights:")
print(percep2.getweights())
print("End bias:")
print(percep2.getbias())
print("")