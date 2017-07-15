import copy, numpy as np

# Compute Sigmoid nonlinearity.
def sigmoid(x):
    output = 1.0 / (1.0 + np.exp(-x))
    return output

# Convert output of Sigmoid function to its derivative.
def sigmoid_output_to_derivative(output):
    return output * (1- output)

# Training dataset generation.
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# Input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# Initialise Neural Network weights.
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# Training logic.
for j in range(10000):
    # Generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # Binary encoding.

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # Binary encoding.

    # True answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # Where we'll store our best guess (binary encoded).
    d = np.zeros_like(c)

    overallError = 0

    l2_deltas = list()
    l1_values = list()
    l1_values.append(np.zeros(hidden_dim))

    # Moving along the positions in the binary encoding.
    for position in range(binary_dim):
        # Generate input and output.
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # Hidden layer (input ~+ prev_hidden)
        l1 = sigmoid(np.dot(X, synapse_0) + np.dot(l1_values[-1], synapse_h))

        # Output layer (new binary representation).
        l2 = sigmoid(np.dot(l1, synapse_1))

        #Did we miss? ... if so, by how much?
        l2_error = y - l2
        l2_deltas.append((l2_error)*sigmoid_output_to_derivative(l2))
        overallError += np.abs(l2_error[0])

        # Decode estimate so we can print it out.
        d[binary_dim - position - 1] = np.round(l2[0][0])

        # Store hidden layer so we can use it in the next timestep.
        l1_values.append(copy.deepcopy(l1))

    future_l1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        l1 = l1_values[-position - 1]
        prev_l1 = l1_values[-position - 2]

        # Error at output layer.
        l2_delta = l2_deltas[-position - 1]
        # Error at hidden layer.
        l1_delta = (future_l1_delta.dot(synapse_h.T) + l2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(l1)

        # Let's update all our weights so we can try again.
        synapse_1_update += np.atleast_2d(l1).T.dot(l2_delta)
        synapse_h_update += np.atleast_2d(prev_l1).T.dot(l1_delta)
        synapse_0_update += X.T.dot(l1_delta)

        future_l1_delta = l1_delta

synapse_0 += synapse_0_update * alpha
synapse_1 += synapse_1_update * alpha
synapse_h += synapse_h_update * alpha

synapse_0_update *= 0
synapse_1_update *= 0
synapse_h_update *= 0
n
# Print out progress.
if( j % 1000 == 0):
    print "Error: " + str(overallError)
    print "Pred: " + str(d)
    print "True: " + str(c)
    out = 0
    for index, x in enumerate(reversed(d)):
        out += x * pow(2, index)
    print str(a_int) + " + " + str(b_int) + " = " + str(out)
    print "-----------"
