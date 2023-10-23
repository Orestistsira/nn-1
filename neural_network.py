import numpy as np

from utils import unpickle


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize the weights and biases for hidden layers and output layer
        self.weights = []
        self.biases = []

        layers = [input_size] + hidden_layers + [output_size]
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
            self.biases.append(np.zeros((1, layers[i])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, x):
        self.layer_outputs = []
        input_data = x
        self.layer_outputs.append(input_data)

        for i in range(len(self.hidden_layers)):
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.layer_outputs.append(a)
            input_data = a

        z = np.dot(input_data, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z)
        self.layer_outputs.append(output)

        return output

    def backward(self, x, y, output):
        error = y - output
        deltas = [error * self.sigmoid_derivative(output)]

        for i in range(len(self.hidden_layers), 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.layer_outputs[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layer_outputs[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += layer.T.dot(delta) * self.learning_rate
            self.biases[i] += delta.sum(axis=0) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            correct = 0
            for i in range(len(X)):
                output = self.feedforward(X[i])
                correct += int(np.argmax(output) == np.argmax(y[i]))
                self.backward(X[i], y[i], output)
            print(f'Epoch {epoch + 1} accuracy: {correct / X.shape[0]:.2f}')

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            output = self.feedforward(X[i])
            predictions.append(output)
        return predictions


def calculate_accuracy(y_test, predictions):
    correct = 0
    for i in range(len(predictions)):
        correct += int(np.argmax(predictions[i]) == np.argmax(y_test[i]))
    return correct / len(y_test)


x_train_1, y_train_1 = unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = unpickle("cifar-10/test_batch")

input_size = 3072
hidden_layers = [20]  # Adjust the hidden layer sizes as needed
output_size = 10

nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate=0.01)
nn.train(x_train, y_train, epochs=3)

predictions = nn.predict(x_test)

accuracy = calculate_accuracy(y_test, predictions)

# Print the accuracy results
print(f"Accuracy: {accuracy:.2f}")
