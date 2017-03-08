"""network_simple2.py
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

network_simple2.py is different from network_simple.py in that it uses the
cross-entropy cost function and it implements L2 regularization with the
hyper-parameter ``lmbda`` passed to SGD.

"""

import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list was
        [2, 3, 1] then it would be a three-layer network, with the first
        layer containing 2 neurons, the second layer 3 neurons, and the third
        layer 1 neuron.  The biases and weights for the network are
        initialized randomly, using a Gaussian distribution with mean 0, and
        variance 1.  Note that the first layer is assumed to be an input
        layer, and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs from later
        layers.

        """
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, iters, eta, lmbda=0.0, test_data=None,
            verbose=False):
        """Train the neural network using stochastic gradient descent.  The
        ``training_data`` is a list of tuples ``(x, y)`` representing the
        training inputs and the desired outputs.  ``iters`` is the number of
        iterations to trian the network on. ``eta`` is the learning rate used
        by the gradient descent algorithm. ``lmbda`` is the parameter to
        control L_2 regularization; bigger ``lmbda`` causes higher
        regularization of parameters. If ``verbose``==True, program will print
        completion statement after every iteration of training network. This
        function returns two lists: ``train_cost`` containing the cost function
        applied to the trianing data after every iteration and
        ``test_accuracy`` containg the accuracy of predictions on the test_data
        after every iteration.  ``test_accuracy`` will be empty if test_data is
        not passed.

        """
        train_cost = []
        test_accuracy = []
        m = len(training_data)
        for i in range(iters):
            for x, y in training_data:
                delta_w, delta_b = self.backprop(x, y)
                self.weights = [(1-eta*(lmbda/m))*w-eta*dw
                                for w, dw in zip(self.weights, delta_w)]
                self.biases = [b-eta*db
                               for b, db in zip(self.biases, delta_b)]
            train_cost.append(self.cost(training_data, lmbda))
            if test_data:
                test_accuracy.append(self.evaluate(test_data))
            if verbose:
                print('Network trained on %d iterations.' % (i+1))
        return train_cost, test_accuracy

    def cost(self, data, lmbda):
        """Returns the cost of the network evaluated on ``data``."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += (-1.0)*np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))
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
