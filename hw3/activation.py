import numpy as np

import layer


class Activation(layer.Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, _):
        return np.multiply(output_gradient, self.activation_prime(self.input))
