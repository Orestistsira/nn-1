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
hidden_layer_size = 50
output_size = 10

dense_layers = [
    DenseLayer(input_size, hidden_layer_size, activation="sigmoid"),
    DenseLayer(hidden_layer_size, output_size, activation="softmax")
]

nn = NeuralNetwork(dense_layers, learn_rate=0.01)
nn.train(x_train, y_train, epochs=10, batch_size=20, validation_data=(x_test, y_test))

accuracy = nn.predict(x_test, y_test)

# Print the accuracy results
print(f"Test accuracy: {accuracy:.2f}")
