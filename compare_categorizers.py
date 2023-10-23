import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
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


x_train, y_train = unpickle("cifar-10/data_batch_1")
x_test, y_test = unpickle("cifar-10/test_batch")

# Initialize and train K-Nearest Neighbors classifiers
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_1.fit(x_train, y_train)
knn_3.fit(x_train, y_train)

# Initialize and train a K-Means model to find the nearest centers
kmeans = KMeans(n_clusters=len(np.unique(y_train)), n_init=10, random_state=0)
kmeans.fit(x_train)

# Predict labels using K-Means for test data
nearest_center_labels = kmeans.predict(x_train)

# Predict labels using K-Nearest Neighbors for test data
knn_1_labels = knn_1.predict(x_train)
knn_3_labels = knn_3.predict(x_train)

# Calculate accuracy for each categorizer
accuracy_nearest_center = accuracy_score(y_test, nearest_center_labels)
accuracy_knn_1 = accuracy_score(y_test, knn_1_labels)
accuracy_knn_3 = accuracy_score(y_test, knn_3_labels)

# Print the accuracy results
print(f"Accuracy (Nearest Center): {accuracy_nearest_center:.2f}")
print(f"Accuracy (1-Nearest Neighbors): {accuracy_knn_1:.2f}")
print(f"Accuracy (3-Nearest Neighbors): {accuracy_knn_3:.2f}")
