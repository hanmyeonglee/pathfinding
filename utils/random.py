from utils.cluster import __select_clustering_func, __select_distance_func

import numpy as np, random


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