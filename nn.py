import numpy as np
import utils
from history import History


class NeuralNetwork:
    def __init__(self, layers, learn_rate=0.01):
        self.learn_rate = learn_rate
        self.layers = layers
        self.hidden_layers_sizes = []
        for i in range(1, len(self.layers)):
            self.hidden_layers_sizes.append(layers[i].input_size)

    def feedforward(self, x):
        y_output = x

        for layer in self.layers:
            y_output = layer.forward(y_output)

        return y_output

    def backward(self, y, y_output):
        # TODO: Add momentum to training
        # Backpropagation
        delta = y_output - y
        for i in range(len(self.layers) - 1, 0, -1):
            # Update layer weights and biases
            layer = self.layers[i]
            layer.backward(delta, self.learn_rate)

            # Update gradients
            prev_activation = self.layers[i - 1].activation
            delta = layer.update_grad(delta, prev_activation)

        self.layers[0].backward(delta, self.learn_rate)

    def train(self, x, y, epochs=3, batch_size=32, validation_data=()):
        history = History()
        history.hyperparams = {
            'num_of_hidden_layers': len(self.layers) - 1,
            'hid_layers_sizes': self.hidden_layers_sizes,
            'learn_rate': self.learn_rate,
            'batch_size': batch_size
        }

        # Divide learning rate with the batch size in order to average out the added gradients
        self.learn_rate /= batch_size
        print('Training...')

        for epoch in range(epochs):
            nr_correct = 0
            loss = []
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

                # Loss / Error calculation
                # e.append(np.sum(1 / len(y_output) * np.sum((y_output - batch_y) ** 2, axis=0)))
                loss.append(np.mean(np.mean(-batch_y * np.log(y_output) - (1 - batch_y) * np.log(1 - y_output), axis=0)))

            # History
            history.loss_history.append(np.mean(loss))

            train_acc = nr_correct / x.shape[0]
            test_acc = 0
            if validation_data:
                test_acc = self.predict(validation_data[0], validation_data[1])

            history.train_acc_history.append(train_acc)
            history.test_acc_history.append(test_acc)

            # Show accuracy for this epoch
            print(f"Epoch {epoch + 1}/{epochs} train accuracy: {train_acc:.2f} - test accuracy: {test_acc:.2f}")

        # Plot the training history
        history.plot_training_history()

    def predict(self, x, y):
        nr_correct = 0
        for img, l in zip(x, y):
            img.shape += (1,)
            l.shape += (1,)

            y_output = self.feedforward(img)
            nr_correct += int(np.argmax(y_output) == np.argmax(l))

        return nr_correct / x.shape[0]
