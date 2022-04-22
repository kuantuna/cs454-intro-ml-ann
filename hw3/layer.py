from typing import Callable
import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class Layer():
    def __init__(self, input_size: int, output_size: int, activation: Callable = sigmoid):
        """Number of input neurons"""
        self.input_size: int = input_size
        """Number of output neurons"""
        self.output_size: int = output_size

        """shape: input_size x output_size (matrix)"""
        self.weights = np.random.randn(input_size, output_size)
        """shape: output_size x 1 (vector)"""
        self.bias = np.random.randn(output_size, 1)

        self.activation: Callable = activation

    def forward(self, input) -> np.ndarray:
        """Expects input to be a numpy array of size input_size x 1 and returns a numpy array of size output_size x 1"""
        return self.activation(np.matmul(self.weights.T, input) + self.bias)

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError()
