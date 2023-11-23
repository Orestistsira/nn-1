from uuid import uuid4

import numpy as np
from matplotlib import pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = data[b'data']
    y = data[b'labels']

    x = x.astype('float32')  # this is necessary for the division below
    x /= 255

    y = to_categorical(y, 10)

    return x, y


def to_categorical(labels, num_classes):
    categorical_labels = np.zeros((len(labels), num_classes))
    categorical_labels[np.arange(len(labels)), labels] = 1
    return categorical_labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def ReLU(x):
    return np.maximum(x, 0)


def ReLU_derivative(x):
    return x > 0


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1. - np.tanh(x) ** 2


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def softmax_derivative(x):
    return np.diagflat(x) - np.dot(x, x.T)


def show_image(x, y, prediction):
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    im_r = x[0:1024].reshape(32, 32)
    im_g = x[1024:2048].reshape(32, 32)
    im_b = x[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.title(f"Label: {classes[np.argmax(y)]} Prediction: {classes[np.argmax(prediction)]}")
    plt.axis('off')
    plt.show()


def plot_training_history(history):
    epochs = list(range(1, len(history.train_acc_history) + 1))
    # Create a single figure with two subplots (2 rows, 1 column)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(epochs, history.train_acc_history, label='Train Accuracy')
    axes[0].plot(epochs, history.test_acc_history, label='Test Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, history.error_history, label='Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Error History')
    axes[1].legend()
    axes[1].grid()

    num_of_hidden_layers = history.hyperparams['num_of_hidden_layers']
    hid_layers_sizes = history.hyperparams['hid_layers_sizes']
    learn_rate = history.hyperparams['learn_rate']
    batch_size = history.hyperparams['batch_size']

    fig.suptitle(f'hidden layers = {num_of_hidden_layers}, hidden layers sizes = {hid_layers_sizes}, learning rate = {learn_rate}, batch size = {batch_size}')
    plt.tight_layout()
    plt.savefig(f'history/history-{num_of_hidden_layers}-{hid_layers_sizes}-{learn_rate}-{batch_size}.jpg')
