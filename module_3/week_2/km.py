import numpy as np
from sklearn.cluster import KMeans

k = 2
data_input = np.array([1.4, 1.0, 1.5, 3.1, 3.8, 4.1])


def example_km():
    # 1. Randomly select centroids
    centroids = data_input[:k]

    limit_assign = 10

    for _ in range(limit_assign):
        # 2. Compute distance
        distances = np.sqrt((data_input.reshape(-1, 1) - centroids)**2)

        # 3. Compute labels
        labels = np.argmin(distances, axis=1)

        # 4. update centroids
        centroids_news = np.array([data_input[labels == i].mean() for i in range(k)])

        # 5. Check stop point
        if np.all(centroids_news == centroids):
            break
        centroids = centroids_news


def example_km_sklearn():
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_input.reshape(-1, 1))
    labels = kmeans.labels_
    print(labels)
    print(kmeans.inertia_)

    for x, label in zip(data_input, labels):
        print(f"Cluster {label}: {x}")


example_km_sklearn()
