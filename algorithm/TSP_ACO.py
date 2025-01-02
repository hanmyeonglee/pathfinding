import numpy as np

def tsp_ant_colony_optimization(
        dmat: np.ndarray, init_path: list[int] | None = None, dist: int | float | None = None,
        n_ants: int = 10, max_iter: int= 100,
        alpha: float = 0.9, beta: float = 1.5, evaporation: float = 0.5, Q: float = 1.0
    ) -> tuple[list, int | float]:
    n = dmat.shape[0]
    pheromone = np.ones(shape=(n, n))
    if init_path is not None:
        init_path = np.array(init_path)
        pheromone[init_path[:-1], init_path[1:]] += Q / dist
        pheromone[init_path[-1], init_path[0]] += Q / dist
    
    best_path = init_path
    best_dist = np.inf

    for iter in range(1, max_iter + 1):
        paths = np.zeros(shape=(n_ants, n), dtype=np.uint16)
        dists = np.zeros(shape=n_ants)
        
        for ant in range(n_ants):
            unvisited = np.ones(n, dtype=np.bool_)
            index = np.arange(n)

            current_point = np.random.randint(n)
            unvisited[current_point] = False
            paths[ant][0] = current_point

            for i in range(1, n):
                unvisited_points = index[unvisited]
                probabilties = pheromone[current_point, unvisited_points] ** alpha / dmat[current_point, unvisited_points] ** beta
                probabilties /= np.sum(probabilties)

                next_point = np.random.choice(unvisited_points, p=probabilties)
                paths[ant][i] = next_point
                dists[ant] += dmat[current_point, next_point]
                unvisited[next_point] = False
                current_point = next_point
            
            if dists[ant] < best_dist:
                best_path = paths[ant]
                best_dist = dists[ant]
            
        pheromone *= evaporation
        for i in range(n_ants):
            pheromone[paths[i, :-1], paths[i, 1:]] += Q / dists[i]
            pheromone[paths[i, -1], paths[i, 0]] += Q / dists[i]

        print(f"\r{iter} / {max_iter} ({iter * 100 / max_iter :.4f}%)", end="")
    
    print()

    return best_path.tolist(), best_dist