from utils.random import make_random_distanceMatrix
from utils.draw import draw_points_withPath
from greedy.TSP_GREEDY import tsp_greedy, k_tsp_greedy
from typing import Callable
from time import time

import numpy as np

def main_kMean_dmat(k: int, solve_func: Callable):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = solve_func(dmat)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_k_tspSolve(k: int, solve_func: Callable):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000))
    points, dmat = dmats[0]

    clusters, pathes, distances = list(), list(), list()
    k_path, k_dist = solve_func(dmat, k=k)

    for path, dist in zip(k_path, k_dist):
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

if __name__ == "__main__":
    main_kMean_dmat(2, tsp_greedy)
    main_k_tspSolve(2, k_tsp_greedy)