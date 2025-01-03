from sklearn.cluster import KMeans
from typing import Callable

import numpy as np, random

def __L1_distance(points: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(points - points[:, None]), axis=-1)

def __L2_distance(points: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(points - points[:, None]) ** 2, axis=-1) ** 0.5

def __k_means(points: np.ndarray, k: int, distance_func: Callable) -> list[np.ndarray]:
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(points)

    ret = []
    indice = np.arange(points.shape[0])
    for label in range(k):
        ret.append(
            points[indice[kmeans.labels_ == label]]
        )
    
    return ret

def __nearest_neighbor(points: np.ndarray, k: int, distance_func: Callable) -> list[np.ndarray]:
    dmat = distance_func(points)
    n = dmat.shape[0]
    unvisited = np.ones(shape=n, dtype=np.bool_)

    start_point = [0 for _ in range(k)]
    indice = np.arange(n)

    clusters = [list((start_point[0], )) for _ in range(k)]
    distances = [0 for _ in range(k)]

    unvisited[start_point[0]] = False
    while unvisited.any():
        if distances[0] > distances[1]: i = 1
        else: i = 0

        start = start_point[i]
        
        target = indice[unvisited]
        end = target[np.argmin(dmat[start, target])]
        
        distances[i] += dmat[start, end]
        start_point[i] = end

        unvisited[end] = False
        clusters[i].append(end)

    clusters = [points[np.array(clusters[i])] for i in range(k)]
    return clusters

def __angle_based_k_means():
    raise NotImplementedError()

def __select_distance_func(distance_type) -> Callable:
    match distance_type:
        case "L1": return __L1_distance
        case "L2": return __L2_distance
        case _: raise Exception("Wrong distance function type.")

def __select_clustering_func(clustering_type) -> Callable:
    match clustering_type:
        case "kmeans": return __k_means
        case "NN": return __nearest_neighbor
        case "angular_kmeans": return __angle_based_k_means
        case _: raise Exception("Wrong clustering function type.")

def make_random_distanceMatrix(
        n: int, max_position: tuple = (10, 10),
        k: int = 1, distance_type: str = "L1",
        clustering_type: str = "kmeans"
    ) -> list[list, np.ndarray]:

    distance_func = __select_distance_func(distance_type)
    clustering_func = __select_clustering_func(clustering_type)

    points = set()
    X, Y = max_position
    for _ in range(n):
        while True:
            point = random.randint(0, X), random.randint(0, Y)
            if point not in points:
                points.add(point)
                break
    
    if k > 1: clusters = clustering_func(np.array(list(points)), k, distance_func)
    else: clusters = [np.array(list(points))]
    dmats = []

    for cluster in clusters:
        dmats.append(
            (cluster.tolist(), distance_func(cluster))
        )

    return dmats