import numpy as np

def __distance(dmat: np.ndarray, path: np.ndarray) -> int | float:
    if len(path.shape) == 1:
        return np.sum(dmat[path[:-1], path[1:]]) + dmat[path[-1], path[0]]
    elif len(path.shape) == 2:
        return np.sum(dmat[path[:, :-1], path[:, 1:]]) + dmat[path[:, -1], path[:, 0]]
    else:
        raise Exception('Wrong path dimension.')


def __2opt(path: np.ndarray) -> np.ndarray:
    n = path.shape[0]
    candidates = np.tile(path, reps=((n - 1) * n // 2, 1))

    row = 0
    for i in range(n - 2):
        for j in range(i + 1, n):
            candidates[row, i:j] = np.flip(candidates[row, i:j])
            row += 1

    return candidates


def tsp_tabu_search(dmat: np.ndarray, path: list[int] | None = None, stop_iter_num: int = 1000) -> tuple[list, int | float]:
    n = dmat.shape[0]
    if path is None:
        path = np.arange(n)
        np.random.shuffle(path)
    else:
        path = np.array(path)

    current_path = path.copy()
    current_dist = __distance(dmat, path)
    best_path = path.copy()
    best_dist = current_dist

    tabu = list()
    max_tabu_size = n * 10

    targets = [(i, j) for i in range(n - 2) for j in range(i + 1, n)]

    iter_num = 0
    while True:
        candidates = __2opt(current_path)
        distances = __distance(dmat, candidates)

        regional_best_path = candidates[0]
        regional_best_dist = distances[0]
        for i, target in enumerate(targets):
            if i == 0: continue
            if distances[i] >= regional_best_dist: continue

            regional_best_path = candidates[i]
            regional_best_dist = distances[i]
            
            if target in tabu: continue

            tabu.append(target)

        if len(tabu) > max_tabu_size:
            tabu = tabu[len(tabu) - max_tabu_size:]

        if regional_best_dist < current_dist:
            current_path = regional_best_path
            current_dist = regional_best_dist

        if current_dist < best_dist:
            best_path = current_path
            best_dist = current_dist
        
        print(f"\r{iter_num} / {stop_iter_num} ({iter_num * 100 / stop_iter_num :.4f}%)", end="")
        iter_num += 1
        if iter_num > stop_iter_num: break

    print()    
    
    return best_path, best_dist