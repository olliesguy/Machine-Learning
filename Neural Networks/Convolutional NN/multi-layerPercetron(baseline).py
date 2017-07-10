import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# Fix random seed for reproducibility.
seed = 7
numpy.random.seed(seed)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training dataset is structured as a 3D array of instance, image wdith, and image height.
# For multi-layer perceptron model we must reduce the images down into a vector of pixels.
# Flatten 28*28 images to a 784 vector for each image.
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Pixel values are gray scale between 0 and 255. Good idea to perform some scaling of input values when using NN models.
# Normalise inputs from 0-255 to 0-1.
X_train = X_train / 255
X_test = X_test / 255

# Finally, output variable is an integer from 0 to 9. This is a multi-class classification problem.
# It is good practise to use a one hot encoding of the class values, transforming the vector of class integers into a binary matrix.
# One hot encode outputs.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Now ready to create a simple NN model with one hidden layer with the same number of neurons as there are inputs (784).
# Define baseline model.
def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # A rectifier activation function is used for the neurons in the hideen layer.
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax')) # A softmax activation function is used on the output layer to turn the outps into probability-like values
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Logarithmic loss is used as the loss function and the ADAM gradient descent algorithm is used to learn the weights.
    return model

# Build the model
model = baseline_model()
#Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
