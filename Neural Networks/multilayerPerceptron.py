# import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 128    # Number of images used in each optimisation step.
nb_classes = 10     # One class per digit.
nb_epoch = 20       # Number of times the whole data is used to learn.

np.random.seed(1)   # for reproducibility

(X_train, y_train), (X_test, y_test) = input_data.read_data_sets('MNIST_data', one_hot=True)

# Flatten the data, MLP doesn't use the 2D structure of the data. 28*28 = 784
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Make the value float in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Display the shapes to check if everything's okay
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices (i.e. one-hot vectors).
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Define the model architecture
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))  # last layer with one output per class.
model.add(Activation('softmax'))  # We want a score similar to a probability for each class.

rms = RMSprop()
# The function to optomise is the cross entropy between the true label and the output (softmax) of the model.
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])

# Make the model learn.
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test))

# Evaluate how the model does on the test set.
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])
