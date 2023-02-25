import numpy as np
import matplotlib.pyplot as plt


def get_distance_between(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x-y)


def compute_centroid_for_points(points: np.ndarray):
    return np.mean(points, axis=0)


class KMeans:
    def __init__(self) -> None:
        self.centroids = None

    # fits the data and resets the centroids
    def fit(self, training_data):
        self.data = training_data
        self.n = training_data.shape[0]
        self.dimension = training_data.shape[1]
        self.centroids = np.empty(self.n)
        self.training_classes = np.empty(self.n)

    # randomly intialize centroids
    def get_random_initializations(self, K):
        centroid_indices = []

        for i in range(K):

            temp_i = -1
            while True:
                temp_i = np.random.randint(0, self.n)
                if not temp_i in centroid_indices:
                    break

            centroid_indices.append(temp_i)

        return np.array([self.data[i] for i in centroid_indices])

    # Compute loss given centroids and centroid classes
    def compute_loss(self, centroids, centroid_classes):

        loss = 0

        for i in range(self.n):

            loss += np.square(
                get_distance_between(self.data[i], centroids[centroid_classes[i]])
            )

        loss /= self.n
        
        return loss

    # train to get a single cluster point
    def _train_single(
        self,
        K: int,
        previous_centroids: np.ndarray = None,
        centroid_classes: np.ndarray = None,
        min_loss: float = 0.002,
    ):

        if previous_centroids is None:
            previous_centroids = self.get_random_initializations(K)

        if centroid_classes is None:
            centroid_classes = np.zeros(self.n, dtype=int)

        for i in range(self.data.shape[0]):

            training_set = self.data[i]

            closest = 0
            closest_dist = get_distance_between(training_set, previous_centroids[0])

            for k in range(1, K):

                temp_dist = get_distance_between(previous_centroids[k], training_set)

                if temp_dist < closest_dist:
                    closest = k
                    closest_dist = temp_dist

            centroid_classes[i] = closest

        for k in range(K):
            points = self.data[centroid_classes == k]
            previous_centroids[k] = compute_centroid_for_points(points)

        current_loss = self.compute_loss(previous_centroids, centroid_classes)
        print(f'current loss: {current_loss}')

        if current_loss <= min_loss:
            return (previous_centroids, current_loss, centroid_classes)
        else:
            return self._train_single(K, previous_centroids, centroid_classes, min_loss)

    # train for all iterations
    def train(self, clusters: int, iterations: int = 100, min_loss: int = 0.002):

        if self.centroids is None:
            raise ValueError("Please fit data first")

        min_loss = 1000000000
        f_centroids = None
        temp_classes = None

        for _ in range(iterations):
            
            print('\ntrying for new model now.')

            temp_centroids, loss, classes = self._train_single(clusters,min_loss=min_loss)

            if loss < min_loss:
                f_centroids = temp_centroids
                temp_classes = classes

        self.centroids = f_centroids
        self.training_classes = temp_classes

    # predict for a single instance
    def predict(self, instance: np.ndarray):

        closest = 0
        closest_dist = get_distance_between(instance, self.centroids[0])

        for k in range(1, self.centroids.shape[0]):
            temp_dist = get_distance_between(self.centroids[k], instance)
            if temp_dist < closest_dist:
                closest = k
                closest_dist = temp_dist

        return self.centroids[closest]
