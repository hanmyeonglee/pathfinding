from utils.random import make_random_distanceMatrix
from utils.draw import draw_points_withPath
from algorithm.TSP_GREEDY import tsp_greedy, k_tsp_greedy
from algorithm.TSP_kOPT import two_opt_sequentialChange, two_opt_randomChange, two_opt_FRLS, two_opt_FSLR
from algorithm.TSP_GA import tsp_genetic_algorithm
from algorithm.TSP_SA import simulated_annealing_inverseOp, simulated_annealing_swapOp
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

def main_tsp2Opt(k: int, optFunc: Callable):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = optFunc(dmat, *tsp_greedy(dmat))
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_tspGA(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, _ = tsp_greedy(dmat)
        path, dist = tsp_genetic_algorithm(
            dmat=dmat, path=path,
            gen_size=256, ggap=0.1, mutation_probability=0.1,
            threshold=10, max_iter=2000,
        )
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tspSA(k: int, mode: str = "inverse"):
    match mode:
        case "inverse":
            SA_func = simulated_annealing_inverseOp
        case "swap":
            SA_func = simulated_annealing_swapOp
        case _:
            raise Exception("Wrong mode.")

    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = SA_func(dmat, *tsp_greedy(dmat))
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

if __name__ == "__main__":
    # 내가 짠 코드는 대칭 dmat이 기준이므로 비대칭 dmat을 사용하려면 나중에 수정이 필요함

    #main_tspGreedy(2)
    #main_tsp2Opt(2, two_opt_sequentialChange)
    #main_tsp2Opt(2, two_opt_randomChange)
    #main_tsp2Opt(2, two_opt_FSLR)
    #main_tsp2Opt(2, two_opt_FRLS)
    #main_tspGA(2)
    #main_tspSA(2, "inverse")
    #main_tspSA(2, "swap")
    #main_tspSA(2, "insert")

    pass