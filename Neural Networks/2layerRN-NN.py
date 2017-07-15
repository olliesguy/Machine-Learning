import numpy as np

# 2 Layer RN Neural Network

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Input dataset
X_train = np.array([[0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            ])

# Output dataset
y_train = np.array([[0, 0, 1, 1]]).T

# Seed RNs
np.random.seed(1)

# Initialise weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):
    # Fordward propagation
    l0 = X_train
    l1 = nonlin(np.dot(l0,syn0))

    # How much did we miss by?
    l1_error = y_train - l1

    # Multiply how much we missed by the derivative of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # Update weights
    syn0 += np.dot(l0.T,li_delta)

print "Output After Training: "
print l1
