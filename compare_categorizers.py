import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = data[b'data']
    y = data[b'labels']

    x = x.astype('float32')  # this is necessary for the division below
    x /= 255

    return x, y


# Load cifar-10 dataset
x_train_1, y_train_1 = unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = unpickle("cifar-10/test_batch")

# Define and train 1-NN classifier
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(x_train, y_train)

# Predict with 1-NN
y_pred_knn_1 = knn_1.predict(x_test)

# Calculate accuracy for 1-NN
accuracy_knn_1 = accuracy_score(y_test, y_pred_knn_1)
print("1-NN Classifier Accuracy:", accuracy_knn_1)

# Define and train 3-NN classifier
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(x_train, y_train)

# Predict with 3-NN
y_pred_knn_3 = knn_3.predict(x_test)

# Calculate accuracy for 3-NN
accuracy_knn_3 = accuracy_score(y_test, y_pred_knn_3)
print("3-NN Classifier Accuracy:", accuracy_knn_3)

# Define and train nearest-center classifier (using K-Means)
kmeans = MiniBatchKMeans(n_clusters=10, n_init=10)
kmeans.fit(x_train)

# Predict with nearest-center classifier
y_pred_nearest_center = kmeans.predict(x_test)

# Calculate accuracy for nearest-center classifier
accuracy_nearest_center = accuracy_score(y_test, y_pred_nearest_center)
print("Nearest-Center Classifier Accuracy:", accuracy_nearest_center)
