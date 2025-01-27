from utils.random import make_random_distanceMatrix, make_random_trashSet
from utils.draw import draw_points_withPath, draw_points_withPath_withStartPoint, draw_realPath
from utils.process import make_distanceMatrix

from TSP.TSP_GREEDY import tsp_greedy
from TSP.TSP_kOPT import two_opt_sequentialChange, two_opt_randomChange, two_opt_FRLS, two_opt_FSLR
from TSP.TSP_GA import tsp_genetic_algorithm
from TSP.TSP_SA import simulated_annealing_inverseOp, simulated_annealing_swapOp
from TSP.TSP_TS import tsp_tabu_search
from TSP.TSP_ACO import tsp_ant_colony_optimization

from TSP_GPU.TSP_GA import tsp_genetic_algorithm_gpu
from TSP_GPU.TSP_ACO import tsp_ant_colony_optimization_gpu

from mTSP.mTSP_GREEDY import k_tsp_greedy
from mTSP.mTSP_GA1 import mtsp_genetic_algorithm_allocation
from mTSP.mTSP_GA2 import mtsp_genetic_algorithm_split

from typing import Callable

import numpy as np, cupy as cp

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
    dmats = make_random_distanceMatrix(1000, max_position=(1000, 1000), k=k)

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
            gen_size=256, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tspGA_with2Opt(k: int, optFunc: Callable):
    dmats = make_random_distanceMatrix(1000, max_position=(5000, 5000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, _ = optFunc(dmat, *tsp_greedy(dmat))
        path, dist = tsp_genetic_algorithm(
            dmat=dmat, path=path,
            gen_size=256, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tsp2Opt_withGA(k: int, optFunc: Callable):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, _ = tsp_greedy(dmat)
        path, dist = tsp_genetic_algorithm(
            dmat=dmat, path=path,
            gen_size=256, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        path, dist = optFunc(dmat, path, dist)
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

def main_tspTS(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_tabu_search(dmat, tsp_greedy(dmat)[0], stop_iter_num=5000)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tspACO(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_ant_colony_optimization(dmat, *tsp_greedy(dmat))
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tspGA_GPU(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, _ = tsp_greedy(dmat)
        path, dist = tsp_genetic_algorithm_gpu(
            dmat=cp.array(dmat), path=path,
            gen_size=2048, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_tspACO_GPU(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k)

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_ant_colony_optimization_gpu(cp.array(dmat), *tsp_greedy(dmat), n_ants=100)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)
    
    draw_points_withPath(clusters, pathes, distances)

def main_clustering_NN(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k, clustering_type="NN")

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, _ = tsp_greedy(dmat)
        path, dist = tsp_genetic_algorithm(
            dmat=dmat, path=path,
            gen_size=256, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        path, dist = two_opt_sequentialChange(dmat, path, dist)
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_clustering_AngleKmeans(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k, clustering_type="angular_kmeans")

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = tsp_greedy(dmat)
        path, dist = two_opt_sequentialChange(dmat, path, dist)
        path, dist = tsp_genetic_algorithm(
            dmat=dmat, path=path,
            gen_size=256, ggap=0.2, mutation_probability=0.05,
            threshold=10, max_iter=5000,
        )
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_clustering_AngleKmeans_doubleKOpt(k: int):
    dmats = make_random_distanceMatrix(100, max_position=(1000, 1000), k=k, clustering_type="angular_kmeans")

    clusters, pathes, distances = list(), list(), list()
    for points, dmat in dmats:
        path, dist = two_opt_sequentialChange(dmat, two_opt_sequentialChange(dmat, *tsp_greedy(dmat)))
        clusters.append(points)
        pathes.append(path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)


def main_mtspNN(k: int):
    points, dmat = make_random_distanceMatrix(100, max_position=(1000, 1000))[0]
    routes, dists = k_tsp_greedy(dmat, k=k)

    clusters, pathes, distances = list(), list(), list()
    for path, dist in zip(routes, dists):
        clusters.append(points)
        pathes.append([0] + path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_mtspGA_allocator(k: int):
    points, dmat = make_random_distanceMatrix(100, max_position=(1000, 1000))[0]
    routes, _ = k_tsp_greedy(dmat, k=k)
    routes, dists = mtsp_genetic_algorithm_allocation(
        k=k,
        dmat=dmat,
        pathes=routes,
        gen_size=256,
        ggap=0.2,
        mutation_probability=0.05,
        threshold=10,
        max_iter=5000,
    )

    clusters, pathes, distances = list(), list(), list()
    for path, dist in zip(routes, dists):
        clusters.append(points)
        pathes.append([0] + path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)

def main_mtspGA_spliter(k: int):
    points, dmat = make_random_distanceMatrix(100, max_position=(1000, 1000))[0]
    routes, _ = k_tsp_greedy(dmat, k=k)
    routes, dists = mtsp_genetic_algorithm_split(
        k=k,
        dmat=dmat,
        pathes=routes,
        gen_size=256,
        ggap=0.2,
        mutation_probability=0.05,
        threshold=10,
        max_iter=5000,
    )

    clusters, pathes, distances = list(), list(), list()
    for path, dist in zip(routes, dists):
        clusters.append(points)
        pathes.append([0] + path)
        distances.append(dist)

    draw_points_withPath(clusters, pathes, distances)
    

if __name__ == "__main__":
    # 내가 짠 코드는 대칭 dmat이 기준이므로 비대칭 dmat을 사용하려면 나중에 수정이 필요함
    
    # main_tspGreedy(2)
    # main_tsp2Opt(2, two_opt_sequentialChange)
    # main_tsp2Opt(2, two_opt_randomChange)
    # main_tsp2Opt(2, two_opt_FSLR)
    # main_tsp2Opt(2, two_opt_FRLS)
    # main_tspGA(2)
    # main_tspGA_with2Opt(2, two_opt_sequentialChange)
    # main_tspGA_with2Opt(2, two_opt_randomChange)
    # main_tspGA_with2Opt(2, two_opt_FSLR)
    # main_tspGA_with2Opt(2, two_opt_FRLS)
    # main_tsp2Opt_withGA(2, two_opt_sequentialChange)
    # main_tsp2Opt_withGA(2, two_opt_randomChange)
    # main_tsp2Opt_withGA(2, two_opt_FSLR)
    # main_tsp2Opt_withGA(2, two_opt_FRLS)
    # main_tspSA(2, "inverse")
    # main_tspSA(2, "swap")
    # main_tspSA(2, "insert")
    # main_tspTS(2)
    # main_tspACO(2)
    # main_tspGA_GPU(2)
    # main_tspACO_GPU(2)

    # main_clustering_NN(2)
    # main_clustering_AngleKmeans(2)

    # main_mtspNN(2)
    # main_mtspGA_allocator(2)
    # main_mtspGA_spliter(2)

    pass