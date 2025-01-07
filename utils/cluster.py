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

def __angle_based_k_means(
        points: np.ndarray,
        k: int,
        distance_func: Callable,
        max_iter: int = 10000,
        threshold: float = 1e-4
    ) -> list[np.ndarray]:
    n, d = points.shape

    dimensional_max = np.max(points, axis=0)
    dimensional_min = np.min(points, axis=0)

    centers = np.random.rand(k, d) * (dimensional_max - dimensional_min) + dimensional_min
    start_point = points[0]
    points = points[1:]
    directional_points = points - start_point # n - 1 * d
    sizeA = np.sum(directional_points ** 2, axis=-1) ** 0.5
    for _ in range(max_iter):
        directional_vectors = centers - start_point
        sizeB = np.sum(directional_vectors ** 2, axis=-1) ** 0.5
        
        similarity = np.dot(directional_points, directional_vectors.transpose()) / (sizeB * sizeA[:, None])
        groups = np.argmax(similarity, axis=-1)

        next_centers = np.zeros((k, d))
        for group_num in range(k):
            group = points[groups == group_num]
            next_centers[group_num, :] = np.sum(group, axis=0) / group.shape[0]

        if (np.abs(centers - next_centers) < threshold).all(): break

        centers = next_centers
    
    return [np.vstack((points[groups == group_num], start_point)) for group_num in range(k)]


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