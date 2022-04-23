from typing import Callable
import numpy as np
np.random.seed(0)


def mse(y_true, y_pred):
    return (np.power(y_true - y_pred, 2)) / 2


def mse_derivative(y_true, y_pred):
    return y_pred - y_true


class Layer():
    def __init__(self, input_size: int, output_size: int):
        """Number of input neurons"""
        self.input_size: int = input_size
        """Number of output neurons"""
        self.output_size: int = output_size

        """shape: output_size x input_size (matrix)"""
        self.weights = np.random.randn(output_size, input_size)
        """shape: output_size x 1 (vector)"""
        self.bias = np.random.randn(output_size, 1)

    def forward(self, inputs) -> np.ndarray:
        """Expects input to be a numpy array of size input_size x 1 and returns a numpy array of size output_size x 1"""
        self.inputs = inputs
        return np.dot(self.weights, inputs) + self.bias

    def backward(self, output_gradient, learning_rate):
        """Gradient of error wrt weights = gradient of error wrt output * input transposed"""
        weights_gradient = np.dot(output_gradient, self.inputs.T)

        """Gradient of error wrt bias = gradient of error wrt output"""
        bias_gradient = output_gradient

        """Gradient of error wrt inputs = weights transposed * gradient of error wrt output"""
        input_gradient = np.dot(self.weights.T, output_gradient)

        """Update weights and bias"""
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


class ActivationLayer():
    def __init__(self, activation: Callable, activation_derivative: Callable):
        """This layer takes in inputs and applies an activation function to them"""
        """so the input and output size are the same"""
        self.activation: Callable = activation
        self.activation_derivative: Callable = activation_derivative

    def forward(self, inputs) -> np.ndarray:
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_gradient, _):
        """Gradient of error wrt inputs = gradient of error wrt output '*' activation derivative of inputs"""
        """ '*' is element-wise multiplication """
        return np.multiply(output_gradient, self.activation_derivative(self.inputs))


class SigmoidActivationLayer(ActivationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_derivative)
