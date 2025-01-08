import numpy as np

def tsp_greedy(dmat: np.ndarray) -> tuple[list, int | float]:
    n = dmat.shape[0]
    visited = np.zeros(shape=n, dtype=np.bool_)

    start = 0
    indice = np.arange(n)

    dist = 0

    path = np.zeros(shape=n, dtype=np.int16)
    i = 0

    while True:
        visited[start] = True
        path[i] = start

        if visited.all(): break
        
        target = indice[visited == False]
        end = target[np.argmin(dmat[start, target])]
        
        dist += dmat[start, end]
        start = end
        i += 1

    dist += dmat[start, 0]
    
    return path.tolist(), dist