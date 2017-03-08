import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, iters, eta, test_data=None):
        for j in range(iters):
            for x, y in training_data:
                delta_w, delta_b = self.backprop(x, y)
                self.weights = [w-eta*dw
                                for w, dw in zip(self.weights, delta_w)]
                self.biases = [b-eta*db
                               for b, db in zip(self.biases, delta_b)]

    def backprop(self, x, y):
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
        delta = (layer_out[-1]-y) * layer_in_prime[-1]
        delta_w[-1] = np.dot(delta, layer_out[-2].transpose())
        delta_b[-1] = delta
        for l in range(2, self.num_layers):
            sp = layer_in_prime[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_w[-l] = np.dot(delta, layer_out[-l-1].transpose())
            delta_b[-l] = delta
        return delta_w, delta_b

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
