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
    e_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return e_x / e_x.sum(axis=0)


def show_image(x, y, prediction):
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.figure()
    im_r = x[0:1024].reshape(32, 32)
    im_g = x[1024:2048].reshape(32, 32)
    im_b = x[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.title(f"Label: {classes[np.argmax(y)]} Prediction: {classes[np.argmax(prediction)]}")
    plt.axis('off')
    plt.show()
