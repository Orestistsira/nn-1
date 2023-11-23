import numpy as np
import utils
from scipy import signal


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_size = None

    def forward(self, input):
        pass

    def backward(self, delta, learning_rate):
        pass

    def grad(self, delta, activation):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation="sigmoid"):
        super().__init__()
        self.input_size = input_size
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


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, kernel_size, num_of_kernels, activation="sigmoid"):
        super().__init__()
        input_height, input_width, input_depth = input_shape
        self.num_of_kernels = num_of_kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (num_of_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (num_of_kernels, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.num_of_kernels):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, delta, learn_rate):
        kernels_gradient = np.zeros(self.kernels_shape)

        for i in range(self.num_of_kernels):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], delta[i], "valid")

        self.kernels -= learn_rate * kernels_gradient
        self.biases -= learn_rate * delta

    def update_grad(self, delta, prev_activation):
        pass


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
