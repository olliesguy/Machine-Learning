# Plot ad hoc MNIST instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot four images as gray scale
plt.subplot(221)
plot.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plot.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plot.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plot.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# Show the plot
plt.show()
