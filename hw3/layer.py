class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return the output of the layer
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update the weights and bias using the output gradient and the learning rate
        pass
