from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_neurons, numbers_of_inputs_per_neuron):
        self.synpatic_weights = 2 * random.random((numbers_of_inputs_per_neuron, number_of_neurons))


class NeuralNetwork():
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weighted.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, X_train, y_train, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network.
            output_from_l1, output_from_l2 = self.think(X_train)

            # Calculate the error for layer 2
            # (The difference between the desired output and the predicted output).
            l2_error = y_train - output_from_l2
            l2_delta = l2_error * self.__sigmoid_derivative(output_from_l2)

            # Calculate the error for layer 1
            # (By looking at the weights in layer 1, we can determine by how much layer 1 contributed to the error in layer 2).
            l1_error = l2_delta.dot(self.l2.synpatic_weights.T)
            l1_delta = l1_error * self.__sigmoid_derivative(output_from_l1)

            # Calculate how much to adjust the weights by
            l1_adjustment = X_train.T.dot(l1_delta)
            l2_adjustment = output_from_l1.T.dot(l2_delta)

            # Adjust the weights.
            self.l1.synpatic_weights += l1_adjustment
            self.l2.synpatic_weights += l2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_l1 = self.__sigmoid(dot(inputs, self.l1.synpatic_weights))
        output_from_l2 = self.__sigmoid(dot(output_from_l1, self.l2.synpatic_weights))
        return output_from_l1, output_from_l2

    # The neural network prints its weights.
    def print_weights(self):
        print("Layer 1 (4 neurons, each with 3 inputs): ")
        print(self.l1.synpatic_weights)
        print("Layer 2 (1 neuron, with 4 inputs):")
        print(self.l2.synpatic_weights)


if __name__ == "__main__":
    # Seed the RN generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    l1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    l2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(l1, l2)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set.
    # We have 7 examples, each consisting of 3 input values and 1 output value.
    X_train = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    y_train = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(X_train, y_train, 60000)

    print("Stage 2) New synaptic weights are training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print(output)
