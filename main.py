from utils.random import make_random_distanceMatrix
from utils.draw import draw_points_withPath
from algorithm.TSP_GREEDY import tsp_greedy, k_tsp_greedy
from algorithm.TSP_kOPT import two_opt_sequentialChange
from typing import Callable
from time import time

import numpy as np

def main_tspGreedy(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_greedy(dmat)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_k_tspGreedy(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000))
    points, dmat = dmats[0]

    clusters, pathes, distances = list(), list(), list()
    k_path, k_dist = k_tsp_greedy(dmat, k=k)

    for path, dist in zip(k_path, k_dist):
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_tsp2Opt(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_greedy(dmat)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

        path, dist = two_opt_sequentialChange(dmat, path, dist)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

if __name__ == "__main__":
    #main_tspGreedy(2)
    main_tsp2Opt(3)