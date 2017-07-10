import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

# Fix random seed for reproducibility.
seed = 7
numpy.random.seed(seed)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training dataset is structured as a 3D array of instance, image wdith, and image height.
# For multi-layer perceptron model we must reduce the images down into a vector of pixels.
# Reshape to be [samples],[pixels],[width],[height].
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

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

# Define baseline model.
def simple_model():
    # Create model
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(1, 28, 28), activation='relu')) # Convolutional layer with 32 feature maps of size 5x5.
    model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling layer taking the max over a 2*2 patches.
    model.add(Dropout(0.2)) # Regularisation layer that randomly excludes 20% of neurons in the layer in order to reducer overfitting.
    model.add(Flatten()) # Converts the 2D matrix to a vector
    model.add(Dense(128, activation='relu')) # A rectifier activation function is used for the neurons in the hideen layer.
    model.add(Dense(num_classes, activation='softmax')) # A softmax activation function is used on the output layer to turn the outps into probability-like values
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Logarithmic loss is used as the loss function and the ADAM gradient descent algorithm is used to learn the weights.
    return model

# Build the model
model = simple_model()
#Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
