import numpy as np

def k_tsp_greedy(dmat: np.ndarray, k: int = 1) -> tuple[list, int | float]:
    n = dmat.shape[0]
    visited = np.zeros(shape=n, dtype=np.bool_)

    start_point = [0 for _ in range(k)]
    indice = np.arange(n)

    distances = [0 for _ in range(k)]

    pathes = [list() for _ in range(k)]

    visited[start_point[0]] = True
    while True:
        for i in range(k):
            start = start_point[i]
            
            target = indice[visited == False]
            end = target[np.argmin(dmat[start, target])]
            
            distances[i] += dmat[start, end]
            start_point[i] = end

            visited[end] = True
            pathes[i].append(end)

            if visited.all(): break
        else:
            continue

        break

    for i in range(k):
        distances[i] += dmat[start_point[i], 0]
    
    return pathes, distances