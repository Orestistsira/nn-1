import time

import numpy as np
import utils
from layer import DenseLayer
from nn import NeuralNetwork

x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = utils.unpickle("cifar-10/test_batch")

input_size = 3072
output_size = 10

hidden_layer_size_1 = 500
hidden_layer_size_2 = 500

dense_layers = [
    DenseLayer(input_size, hidden_layer_size_1, activation="sigmoid"),
    DenseLayer(hidden_layer_size_1, hidden_layer_size_2, activation="sigmoid"),
    DenseLayer(hidden_layer_size_2, output_size, activation="softmax")
]

start_time = time.time()
nn = NeuralNetwork(dense_layers, learn_rate=0.01)
nn.train(x_train, y_train, epochs=30, batch_size=50, validation_data=(x_test, y_test))
print('Model successfully trained in %.2fs' % (time.time() - start_time))

accuracy = nn.predict(x_test, y_test, show_image=False)

# Print the accuracy results
print(f"Test accuracy: {accuracy:.2f}")
