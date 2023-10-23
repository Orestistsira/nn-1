import numpy as np
import matplotlib.pyplot as plt
import utils


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, learn_rate=0.01):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learn_rate = learn_rate

        # Initialize the weights and biases for hidden layers and output layer
        self.w_i_h = np.random.rand(hidden_layer_size, input_size) - 0.5
        self.w_h_o = np.random.rand(output_size, hidden_layer_size) - 0.5
        self.b_i_h = np.zeros((hidden_layer_size, 1))
        self.b_h_o = np.zeros((output_size, 1))

        self.nr_correct = 0
        self.e = 0

    def feedforward(self, x):
        # Forward propagation input -> hidden
        u_1 = self.b_i_h + self.w_i_h.dot(x)
        y_1 = utils.sigmoid(u_1)

        # Forward propagation hidden -> output
        u_2 = self.b_h_o + self.w_h_o.dot(y_1)
        y_2 = utils.softmax(u_2)

        return y_1, y_2

    def backward(self, x, y, y_1, y_2):
        # Cost / Error calculation
        self.e = 1 / len(y_2) * np.sum((y_2 - y) ** 2, axis=0)
        self.nr_correct += int(np.argmax(y_2) == np.argmax(y))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = y_2 - y
        self.w_h_o += -self.learn_rate * delta_o.dot(np.transpose(y_1))
        self.b_h_o += -self.learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(self.w_h_o).dot(delta_o) * utils.sigmoid_derivative(y_1)
        self.w_i_h += -self.learn_rate * delta_h.dot(np.transpose(x))
        self.b_i_h += -self.learn_rate * delta_h

    def train(self, x, y, epochs=3):
        self.nr_correct = 0
        print('Training...')
        # TODO: Train in batches
        for epoch in range(epochs):
            for img, l in zip(x, y):
                img.shape += (1,)
                l.shape += (1,)

                y_1, y_2 = self.feedforward(img)
                self.backward(img, l, y_1, y_2)

            # Show accuracy for this epoch
            print(f"Epoch {epoch + 1} accuracy: {self.nr_correct / x_train.shape[0]:.2f}")
            self.nr_correct = 0

    def predict(self, x, y):
        print('Testing...')
        for img, l in zip(x, y):
            img.shape += (1,)
            l.shape += (1,)

            _, y_2 = self.feedforward(img)

            self.nr_correct += int(np.argmax(y_2) == np.argmax(l))

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
hidden_layer_size = 50
output_size = 10

nn = NeuralNetwork(input_size, hidden_layer_size, output_size, learn_rate=0.01)
nn.train(x_train, y_train, epochs=3)

x_test, y_test = utils.unpickle("cifar-10/test_batch")
nn.predict(x_test, y_test)
