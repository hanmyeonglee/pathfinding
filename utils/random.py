from sklearn.cluster import KMeans

import numpy as np, random

def L1_distance(points: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(points - points[:, None]), axis=-1)

def L2_distance(points: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(points - points[:, None]) ** 2, axis=-1) ** 0.5

def k_means(points: np.ndarray, k: int) -> list[np.ndarray]:
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(points)

    ret = []
    indice = np.arange(points.shape[0])
    for label in range(k):
        ret.append(
            points[indice[kmeans.labels_ == label]]
        )
    
    return ret

def make_random_distanceMatrix(
        n: int, max_position: tuple = (10, 10),
        k: int = 1, distance_type: str = "L1"
    ) -> list[list, np.ndarray]:

    if distance_type == "L1": distance_func = L1_distance
    elif distance_type == "L2": distance_func = L2_distance
    else: raise Exception("Wrong distance function type.")

    points = set()
    X, Y = max_position
    for _ in range(n):
        while True:
            point = random.randint(0, X), random.randint(0, Y)
            if point not in points:
                points.add(point)
                break

    clusters = k_means(np.array(list(points)), k)
    dmats = []

    for cluster in clusters:
        dmats.append(
            (cluster.tolist(), distance_func(cluster))
        )

    return dmats