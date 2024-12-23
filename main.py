from utils.random import make_random_distanceMatrix
from utils.draw import draw_points_withPath
from greedy.TSP_GREEDY import tsp_greedy
from typing import Callable
from time import time

import numpy as np

def main(solve_func: Callable):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=2)

    clusters, pathes, distances, labels = list(), list(), list(), list()
    for label, (points, dmat) in enumerate(dmats):
        path, dist = solve_func(dmat)

        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
        labels.append(label)

    draw_points_withPath(clusters, pathes, distances, labels)

if __name__ == "__main__":
    main(tsp_greedy)