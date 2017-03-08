"""network.py
~~~~~~~~~~

Some code to implement the stochastic gradient descent learning algorithm for
a feedforward neural network.  Gradients are calculated using
backpropagation.  Note that I have focused on making the code simple, easily
readable, and easily modifiable.  It is not optimized, and omits many
desirable features.

This code has been modified for a talk at Py Data Denver by Ryan Arredondo.
The original code was written by Michael Nielsen and can be found at:
https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src

In this simple implementation of a neural network, the user can control the
number of layers and the size of each layer (see __init__ docstring), the
learning rate ``eta`` passed to SGD, and the number iterations ``iters`` to
train the network on (also passed to SGD).

network_simple2.py was different from network_simple.py in that it used the
cross-entropy cost function and it implemented L2 regularization with the
hyper-parameter ``lmbda`` passed to SGD. These changes are also present in
network_simple3.py.

network_simple3.py is different from network_simple2.py in that it uses the
mini-batch gradient descent. The hyper-parameter ``batch_size`` can be passed
to SGD. There is also an option to initialize the weights of the network
to have a lower variance using the option ``tight_weights``; using this option
will speed up learning.

"""

import numpy as np


class Network(object):

    def __init__(self, sizes, tight_weights=False):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list was
        [2, 3, 1] then it would be a three-layer network, with the first
        layer containing 2 neurons, the second layer 3 neurons, and the third
        layer 1 neuron.  The biases and weights for the network are
        initialized randomly, using a Gaussian distribution with mean 0, and
        variance 1 (if ``tight_weights``==False, otherwise, intializes weights
        with variance 1/n_in where n_in is the number input neurons to the
        weights.  Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later layers.

        """
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if tight_weights:
            self.weights = [np.random.randn(y, x)/np.sqrt(x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights =[np.random.randn(y, x)
                           for x, y in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, iters, eta, lmbda=0.0, batch_size=1,
            test_data=None, verbose=False):
        """Train the neural network using stochastic gradient descent.  The
        ``training_data`` is a list of tuples ``(x, y)`` representing the
        training inputs and the desired outputs.  ``iters`` is the number of
        iterations to trian the network on. ``eta`` is the learning rate used
        by the gradient descent algorithm. ``lmbda`` is the parameter to
        control L_2 regularization; bigger ``lmbda`` causes higher
        regularization of parameters. ``batch_size`` sets the size of
        mini-batches to use in the update step of stochastic gradient
        descent. If ``verbose``==True, program will print completion
        statement after every iteration of training network. This function
        returns two lists: ``train_cost`` containing the cost function
        applied to the trianing data after every iteration and
        ``test_accuracy`` containg the accuracy of predictions on the
        test_data after every iteration.  ``test_accuracy`` will be empty if
        test_data is not passed.

        """
        train_cost = []
        test_accuracy = []
        m = len(training_data)
        for i in range(iters):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                            for k in range(0, m, batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta, lmbda, m)
            train_cost.append(self.cost(training_data, lmbda))
            if test_data:
                test_accuracy.append(self.evaluate(test_data))
            if verbose:
                print('Network trained on %d iterations.' % (i+1))
        return train_cost, test_accuracy

    def update(self, mini_batch, eta, lmbda, m):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and ``m``
        is the total size of the training data set.

        """
        batch_delta_w = [np.zeros(w.shape) for w in self.weights]
        batch_delta_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_w, delta_b = self.backprop(x, y)
            batch_delta_w = [dw+bdw for dw, dgw in zip(delta_w, batch_delta_w)]
            batch_delta_b = [db+bdb for db, bdb in zip(delta_b, batch_delta_b)]
        self.weights = [(1-eta*(lmbda/m))*w-eta*bdw
                        for w, bdw in zip(self.weights, batch_delta_w)]
        self.biases = [b-eta*bdb
                       for b, bdb in zip(self.biases, batch_delta_b)]


    def cost(self, data, lmbda):
        """Returns the cost of the network evaluated on ``data``."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += -1.0*np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))
        cost += 0.5*lmbda*sum(np.sum(w**2) for w in self.weights)
        return (1.0/len(data))*cost

    def evaluate(self, data):
        """Returns number of correct predictions by network on ``data``"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        """Return a tuple ``(delta_w, delta_b)`` representing the
        gradient for the cost function C_x.  ``delta_w`` and
        ``delta_b`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        # Feed it forward
        layer_out = [x]  # Stores output from all layers
        layer_in_prime = []  # Stores sigmoid' of input to all layers
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, layer_out[-1])+b  # Get input to next layer
            layer_out.append(sigmoid(z))
            layer_in_prime.append(sigmoid_prime(z))
        # Take it back
        delta = (layer_out[-1]-y)
        delta_w[-1] = np.dot(delta, layer_out[-2].transpose())
        delta_b[-1] = delta
        for l in range(2, self.num_layers):
            sp = layer_in_prime[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_w[-l] = np.dot(delta, layer_out[-l-1].transpose())
            delta_b[-l] = delta
        return delta_w, delta_b


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
