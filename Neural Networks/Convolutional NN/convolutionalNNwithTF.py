"""
Based on the tflearn Bird Classifier
"""

from __future__ import division, print_function, absolute_import

# Import tflearn and some helpful packages.
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Load the dataset
X_train, y_train, X_test, y_test = pickle.load(open("full_dataset.pkl", "rb"))
# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)

# Make sure the image is normalised.
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating, and lurring the images in our dataset.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftrigh()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:
# Input is a 32x32 image with 3 colour channels (red, green, and blue).
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

network = conv_2d(network, 32, 2, activation='relu') # Step 1: Convolution
network = max_pool_2d(network, 2) # Step 2: Max pooling
network = conv_2d(network, 64, 3, activation='relu') # Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu') # Step 4: Convolution yet again
network = max_pool_2d(network, 2) # Step 5: Max pooling again
network = fully_connected(network, 512, activation='relu') # Step 6: Fully-connected 512 node NN
network = dropout(network, 0.5) # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = fully_connected(network, 2, activation='softmax')# Step 8: Fully-connected NN with two outputs (0 = False = isn't a bird, 1 = True = is a bird) to make the final prediction

# Tell tflearn how we want to train the network.
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Wrap the network in a model object.
model = tflearn.DNN(network=, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')

# Train it for 100 training passes and monitor it as it goes.
model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=(X_test, y_test), show_metrics=True, batch_size=96, snapshot_epoche=True, run_id='bird-classifier')

# Save model when training is complete to a file.
model.save("bird-classifier.tfl")
print("Network trained and saved as bird-classifier.tfl!")
