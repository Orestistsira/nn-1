import numpy as np
import utils


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, delta, learning_rate):
        pass

    def grad(self, delta, activation):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation="sigmoid"):
        super().__init__()
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input):
        self.input = input
        return apply_activation(self.activation, self.bias + self.weights.dot(self.input))

    def backward(self, delta, learn_rate):
        weights_grad = delta.dot(self.input.T)

        self.weights -= learn_rate * weights_grad
        self.bias -= learn_rate * delta.sum(axis=1, keepdims=True)

    def update_grad(self, delta, prev_activation):
        delta = self.weights.T.dot(delta) * apply_activation_derivative(prev_activation, self.input)
        return delta


def apply_activation(activation, u):
    if activation == "sigmoid":
        return utils.sigmoid(u)
    elif activation == "relu":
        return utils.ReLU(u)
    elif activation == "tanh":
        return utils.tanh(u)
    elif activation == "softmax":
        return utils.softmax(u)
    else:
        raise ValueError(f'There is no activation function = {activation}')


def apply_activation_derivative(activation, u):
    if activation == "sigmoid":
        return utils.sigmoid_derivative(u)
    elif activation == "relu":
        return utils.ReLU_derivative(u)
    elif activation == "tanh":
        return utils.tanh_derivative(u)
    elif activation == "softmax":
        return utils.softmax_derivative(u)
    else:
        raise ValueError(f'There is no activation function derivative = {activation}')
