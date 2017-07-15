
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
*X_train* = an array of feature inputs that are used for training the desired algorithm. Typically a subset of a larger dataset.
*y_train* = the matching outputs for the training input data.
*X_test* = an array of test variables (from the same dataset as X_train), which are used to validate the accuracy of the algorithms on "unseen" data.
*y_test* = corresponding true output values for X_test variables.
*l#* = layer number, where # equals any positive integer. Used in Neural Networks.
*cm* = confusion matrix (or error matrix), a visualisation of the performance on an ML algorithm. Example layout below.

														| True Condition								|
					| --------------------------------- | --------------------------------------------- |
					| Total Population					| Condition Positive	| Condition Negative	|
------------------- | --------------------------------- | --------------------- | --------------------- |
Predicted condition	| Predicted condition (positive)	| True Positive			| False positive		|
					| --------------------------------- | --------------------- | --------------------- |
					| Predicted condition (negative)	| False negative		| True negative			|

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
1. Iris
2. MNIST
3. Custom

## Resources:
To be added in due time.
