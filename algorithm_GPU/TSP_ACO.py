import cupy as cp, cupyx as cpx

def tsp_ant_colony_optimization_gpu(
        dmat: cp.ndarray, init_path: list[int] | None = None, dist: int | float | None = None,
        n_ants: int = 10, max_iter: int= 100,
        alpha: float = 0.9, beta: float = 1.5, evaporation: float = 0.5, Q: float = 1.0
    ) -> tuple[list, int | float]:
    n = dmat.shape[0]
    pheromone = cp.ones(shape=(n, n))
    if init_path is not None:
        init_path = cp.array(init_path)
        pheromone[init_path[:-1], init_path[1:]] += Q / dist
        pheromone[init_path[-1], init_path[0]] += Q / dist
    
    best_path = init_path
    best_dist = cp.inf

    index = cp.arange(n_ants)
    for iter in range(1, max_iter + 1):
        paths = cp.zeros(shape=(n_ants, n), dtype=cp.uint16)
        unvisited = cp.ones((n_ants, n), dtype=cp.bool_)
        
        current_points = cp.random.randint(0, n, size=n_ants)
        paths[:, 0] = current_points
        unvisited[index, current_points] = False
        for i in range(1, n):
            unvisited_points_x, unvisited_points_y = cp.where(unvisited)
            unvisited_points_x = unvisited_points_x.reshape((n_ants, n - i))
            unvisited_points_y = unvisited_points_y.reshape((n_ants, n - i))

            start_points = cp.tile(current_points, reps=(n - i, 1)).transpose()
            roulette = (pheromone[start_points, unvisited_points_y] ** alpha) / (dmat[start_points, unvisited_points_y] ** beta)
            roulette = cp.cumsum(roulette, axis=-1)

            mx_each_ants = cp.max(roulette, axis=-1)
            randnums = cp.random.rand(n_ants) * mx_each_ants

            local_current_points = cp.sum(roulette <= randnums[:, None], axis=-1)
            current_points = unvisited_points_y[cp.arange(n_ants), local_current_points]
            paths[:, i] = current_points
            unvisited[index, current_points] = False

        dists = cp.sum(dmat[paths[:, :-1], paths[:, 1:]], axis=-1) + dmat[paths[:, -1], paths[:, 0]]
        min_dist_index = cp.argmin(dists)
        if dists[min_dist_index] < best_dist:
            best_path = paths[min_dist_index]
            best_dist = dists[min_dist_index]

        pheromone *= evaporation
        cpx.scatter_add(pheromone, (paths[:, :-1], paths[:, 1:]), Q / dists[:, None])
        cpx.scatter_add(pheromone, (paths[:, -1], paths[:, 0]), Q / dists)

        print(f"\r{iter} / {max_iter} ({iter * 100 / max_iter :.4f}%)", end="")
    
    print()

    """ for iter in range(1, max_iter + 1):
        paths = cp.zeros(shape=(n_ants, n), dtype=cp.uint16)
        dists = cp.zeros(shape=n_ants)
        
        for ant in range(n_ants):
            unvisited = cp.ones(n, dtype=cp.bool_)
            index = cp.arange(n)

            current_point = cp.random.randint(n - 1)
            unvisited[current_point] = False
            paths[ant][0] = current_point

            for i in range(1, n):
                unvisited_points = index[unvisited]
                probabilties = pheromone[current_point, unvisited_points] ** alpha / dmat[current_point, unvisited_points] ** beta
                probabilties /= cp.sum(probabilties)

                next_point = cp.random.choice(unvisited_points, p=probabilties)
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
    
    print() """

    return best_path.tolist(), best_dist