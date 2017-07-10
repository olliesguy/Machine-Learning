# Run terminal or command line: $ pip3 (or pip) install pybrain

from pybrain.datasets.classification import ClassificationDataSet
# Line 6 can be replaced with the algorithm of choice...
# e.g. from pybrain.optimization.hillclimber import HillClimber
from pybrain.optimization.populationbased.ga import GA
from pybrain.tools.shortcuts import buildNetwork

# Create dataset
d = ClassificationDataSet(2)
d.addSample([181, 80], [1])
d.addSample([177, 70], [1])
d.addSample([160, 60], [0])
d.addSample([154, 54], [0])
d.setField('class', [ [0.], [1.], [1.], [0.]])

nn = buildNetwork(2, 3, 1)

# d.evaluateModuleMSE takes nn as its first and only argument.
ga = GA(d.evaluateModuleMSE, nn, minimise=True)

for i in range(100):
    nn = ga.learn(0)[0]

print nn.activate([181, 80])
