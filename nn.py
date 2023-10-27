import numpy as np
import matplotlib.pyplot as plt
import utils
from layer import DenseLayer


class NeuralNetwork:
    def __init__(self, layers, learn_rate=0.01):
        self.learn_rate = learn_rate
        self.layers = layers
        self.e = 0

    def feedforward(self, x):
        y_output = x

        for layer in self.layers:
            y_output = layer.forward(y_output)

        return y_output

    def backward(self, y, y_output):
        # Cost / Error calculation
        self.e = 1 / len(y_output) * np.sum((y_output - y) ** 2, axis=0)

        # Backpropagation
        delta = y_output - y
        for i in range(len(layers) - 1, 0, -1):
            # Update layer weights and biases
            layer = layers[i]
            layer.backward(delta, self.learn_rate)

            # Update gradients
            prev_activation = layers[i - 1].activation
            delta = layer.update_grad(delta, prev_activation)

        layers[0].backward(delta, self.learn_rate)

    def train(self, x, y, epochs=3, batch_size=32, validation_data=()):
        self.learn_rate /= batch_size
        print('Training...')
        for epoch in range(epochs):
            nr_correct = 0
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
                y_output = self.feedforward(batch_x)
                nr_correct += np.sum(np.argmax(y_output, axis=0) == np.argmax(batch_y, axis=0))
                self.backward(batch_y, y_output)

            test_acc = nr_correct / x_train.shape[0]
            val_acc = 0
            if validation_data:
                val_acc = self.predict(validation_data[0], validation_data[1])

            # Show accuracy for this epoch
            print(f"Epoch {epoch + 1}/{epochs} test accuracy: {test_acc:.2f} - val accuracy: {val_acc:.2f}")

    def predict(self, x, y):
        nr_correct = 0
        for img, l in zip(x, y):
            img.shape += (1,)
            l.shape += (1,)

            y_output = self.feedforward(img)
            nr_correct += int(np.argmax(y_output) == np.argmax(l))

        return nr_correct / x_test.shape[0]


x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = utils.unpickle("cifar-10/test_batch")

input_size = 3072
hidden_layer_size = 50
output_size = 10

layers = [
    DenseLayer(input_size, hidden_layer_size, activation="sigmoid"),
    DenseLayer(hidden_layer_size, output_size, activation="softmax")
]

nn = NeuralNetwork(layers, learn_rate=0.01)
nn.train(x_train, y_train, epochs=10, batch_size=10, validation_data=(x_test, y_test))

accuracy = nn.predict(x_test, y_test)

# Print the accuracy results
print(f"Test accuracy: {accuracy:.2f}")
