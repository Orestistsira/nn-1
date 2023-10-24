import numpy as np
import matplotlib.pyplot as plt
import utils


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size, hidden_activ, learn_rate=0.01):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.hidden_activ = hidden_activ

        self.num_hidden_layers = len(hidden_layer_sizes)

        # Initialize the weights and biases for hidden layers and output layer
        self.hidden_w = [np.random.rand(hidden_layer_sizes[0], input_size) - 0.5]
        self.output_w = np.random.rand(output_size, hidden_layer_sizes[-1]) - 0.5
        self.hidden_b = [np.zeros((hidden_layer_sizes[0], 1))]
        self.output_b = np.zeros((output_size, 1))

        for i in range(1, self.num_hidden_layers):
            self.hidden_w.append(np.random.rand(hidden_layer_sizes[i], hidden_layer_sizes[i - 1]) - 0.5)
            self.hidden_b.append(np.zeros((hidden_layer_sizes[i], 1)))

        self.nr_correct = 0
        self.e = 0

    def hidden_activation(self, u):
        if self.hidden_activ == "sigmoid":
            return utils.sigmoid(u)
        elif self.hidden_activ == "relu":
            return utils.ReLU(u)

    def hidden_activation_derivative(self, u):
        if self.hidden_activ == "sigmoid":
            return utils.sigmoid_derivative(u)
        elif self.hidden_activ == "relu":
            return utils.ReLU_derivative(u)

    def feedforward(self, x):
        # Forward propagation input -> hidden
        y_layers = [self.hidden_activation(self.hidden_b[0] + self.hidden_w[0].dot(x))]

        for i in range(1, self.num_hidden_layers):
            y_layers.append(self.hidden_activation(self.hidden_b[i] + self.hidden_w[i].dot(y_layers[i - 1])))

        # Forward propagation hidden -> output
        u_2 = self.output_b + self.output_w.dot(y_layers[-1])
        y_output = utils.softmax(u_2)

        return y_layers, y_output

    def backward(self, x, y, y_layers, y_output):
        # Cost / Error calculation
        self.e = 1 / len(y_output) * np.sum((y_output - y) ** 2, axis=0)
        self.nr_correct += np.sum(np.argmax(y_output, axis=0) == np.argmax(y, axis=0))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = y_output - y
        self.output_w += -self.learn_rate * delta_o.dot(np.transpose(y_layers[-1]))
        self.output_b += -self.learn_rate * delta_o.sum(axis=1, keepdims=True)

        # Backpropagation hidden -> input (activation function derivative)
        delta_layers = [np.zeros_like(layer) for layer in y_layers]
        delta_layers[-1] = np.transpose(self.output_w).dot(delta_o) * self.hidden_activation_derivative(y_layers[-1])

        for i in range(self.num_hidden_layers - 2, -1, -1):
            delta_layers[i] = np.transpose(self.hidden_w[i + 1]).dot(delta_layers[i + 1]) * ...
            self.hidden_activation_derivative(y_layers[i])

        self.hidden_w[0] += -self.learn_rate * delta_layers[0].dot(np.transpose(x))
        self.hidden_b[0] += -self.learn_rate * delta_layers[0].sum(axis=1, keepdims=True)

        for i in range(1, self.num_hidden_layers):
            self.hidden_w[i] += -self.learn_rate * delta_layers[i].dot(np.transpose(y_layers[i - 1]))
            self.hidden_b[i] += -self.learn_rate * delta_layers[i].sum(axis=1, keepdims=True)

    def train(self, x, y, epochs=3, batch_size=32):
        self.nr_correct = 0
        self.learn_rate /= batch_size
        print('Training...')
        # TODO: Train model in batches
        for epoch in range(epochs):
            # Shuffle the data for each epoch to introduce randomness
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for batch_start in range(0, len(x_shuffled), batch_size):
                batch_end = batch_start + batch_size
                batch_x = x_shuffled[batch_start:batch_end]
                batch_y = y_shuffled[batch_start:batch_end]

                batch_x = batch_x.reshape((batch_x.shape[0], -1)).T  # Reshape the batch data
                batch_y = batch_y.T

                # Forward and backward pass for the batch
                y_layers, y_output = self.feedforward(batch_x)
                self.backward(batch_x, batch_y, y_layers, y_output)

            # Show accuracy for this epoch
            print(f"Epoch {epoch + 1}/{epochs} accuracy: {self.nr_correct / x_train.shape[0]:.2f}")
            self.nr_correct = 0

    def predict(self, x, y):
        print('Testing...')
        for img, l in zip(x, y):
            img.shape += (1,)
            l.shape += (1,)

            _, y_output = self.feedforward(img)

            self.nr_correct += int(np.argmax(y_output) == np.argmax(l))

        # Print the accuracy results
        print(f"Test accuracy: {self.nr_correct / x_test.shape[0]:.2f}")


x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

input_size = 3072
hidden_layer_sizes = [50]
hidden_activ = "sigmoid"
output_size = 10

nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size, hidden_activ, learn_rate=0.01)
nn.train(x_train, y_train, epochs=10, batch_size=10)

x_test, y_test = utils.unpickle("cifar-10/test_batch")
nn.predict(x_test, y_test)
