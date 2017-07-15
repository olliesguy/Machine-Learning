
# Intro to Machine-Learning
In my attempt to rapidly understand and apply machine learning techniques to my MSc individual thesis, I have collated a number of tutorial codes (in Python) together.

This folder encompasses a variety of simple and applied machine learning algorithms. At the time of writing this includes:
* Logic Regression
* Random Forest
* Naive-Bayes
* K-Nearest Neighbors
* Decision Tree
* Neural Networks
	- Artificial
	- Recurrent
	- Convolutional

An effort has been made to standardise variable names and comments across files to aid with learning different algorithms.

## Common Variables:
```python
X_train = an array of feature inputs that are used for training the desired algorithm. Typically a subset of a larger dataset.
y_train = the matching outputs for the training input data.
X_test = an array of test variables (from the same dataset as X_train), which are used to validate the accuracy of the algorithms on "unseen" data.
y_test = corresponding true output values for X_test variables.
lZ = layer number, where Z equals any positive integer. Used in Neural Networks.
cm = confusion matrix (or error matrix), a visualisation of the performance on an ML algorithm.
```

## Library Requirements:
+ numpy
+ scipy
+ tensorflow
+ tflearn
+ sklearn
+ keras
+ pybrain
+ lasagne

## Datasets:
1. MNIST
2. Iris
3. Custom copies

## Resources:
* Siraj Raval's [YouTube channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
* Brandon Rohrer's [YouTube videos](https://www.youtube.com/user/BrandonRohrer) - more for explanation than coded examples.
* Per Harald Borgen - ML in a [Week](https://medium.com/learning-new-stuff/machine-learning-in-a-week-a0da25d59850) or [Year](https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c)
* Andrew Ng [Machine Learning at Stanford](https://www.coursera.org/learn/machine-learning) (MOOC via Coursera)
* Adam Geitgey's [Machine Learning is Fun](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471) (series of Medium posts)
* Andrew Trask's [A NN in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/) and Milo Spencer-Harper's [variant](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)
* Jason Brownlee's [overview of ML topics](http://machinelearningmastery.com/start-here/) (free blog/tutorials and paid training)
* Arun Agrahri's far more [substantiative resource list](https://hackernoon.com/index-of-best-ai-machine-learning-resources-71ba0c73e34d)
Welcome any further suggestions...
