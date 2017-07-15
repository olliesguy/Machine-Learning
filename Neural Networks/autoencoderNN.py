import tensorflow as tf
import numpy as np
import input_data

# Hyperparameters
mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3

# Create node for input data.
X = tf.placeholder("float", [None, n_visible], name='X')

# Create node for corruption mask.
mask = tf.placeholder("float", [None, n_visible], name='mask')

# Create nodes for hidden variables.
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden], minval=-W_init_max, maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W) # Tied weights between encoder and decoder.
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')

def mode(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X # Corrupted X

    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b) # Hidden state.
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime) # Reconstructed input.
    return Z

# Build model graph.
Z = model(X, mask, W, b, W_prime, b_prime)

# Create cost function.
cost = tf.reduce_sum(tf.pow(X - Z, 2)) # Minimise squared error.
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost) # Construct an optimiser with gradient decent.

# Load MNIST data.
mnist = input_data.read_date_sets("MNIST_date/", one_hot=True)
X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session.
with tf.Session() as sess:
    # Initialise all variables.
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train), 128)):
            input_ = X_train[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

            mask_np = np.random.binomial(1, 1 - corruption_level, X_test.shape)
            print(i, sess.run(cost, feed_dict={X: X_test, mask: mask_np}))
