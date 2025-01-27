from math import exp as e, log

import numpy as np, random

def __calculate_distance(dmat: np.ndarray, path: np.ndarray) -> int | float:
    return np.sum(dmat[path[:-1], path[1:]]) + dmat[path[-1], path[0]]

def __selection(n: int) -> tuple[int, int]:
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i

    return i, j

def simulated_annealing_inverseOp(
        dmat: np.ndarray, path: list[int], dist: int | float,
        initial_temperature=1000, cooling_rate=0.995, min_temperature=1e-3, max_iterations=1000
    ) -> tuple[list, int | float]:
    n = len(path)

    current_path = path
    current_dist = dist

    best_path = path
    best_dist = dist

    T = initial_temperature

    total_iter = int(log(min_temperature / initial_temperature, cooling_rate)) + 1
    iter = 0

    while T > min_temperature:
        for _ in range(max_iterations):
            i, j = __selection(n)

            start1, start2 = current_path[i - 1], current_path[j - 1]
            end1, end2 = current_path[i], current_path[j]

            temp_dist =   current_dist \
                        + (dmat[start1, start2] + dmat[end1, end2]) \
                        - (dmat[start1, end1] + dmat[start2, end2])
            
            if not (temp_dist < current_dist or random.random() < e((current_dist - temp_dist) / T)):
                continue

            current_path = current_path[ : i] + current_path[i : j][::-1] + current_path[j : ]
            current_dist = temp_dist

            if current_dist < best_dist:
                best_path = current_path
                best_dist = current_dist
        
        T *= cooling_rate
        iter += 1

        print(f"\r{iter} / {total_iter} ({iter * 100 / total_iter :.4f}%)", end="")

    print()

    return best_path, best_dist


def simulated_annealing_swapOp(
        dmat: np.ndarray, path: list[int], dist: int | float,
        initial_temperature=1000, cooling_rate=0.995, min_temperature=1e-3, max_iterations=1000
    ) -> tuple[list, int | float]:
    n = len(path)
    path = np.array(path)

    current_path = path.copy()
    current_dist = dist

    best_path = path.copy()
    best_dist = dist

    T = initial_temperature

    total_iter = int(log(min_temperature / initial_temperature, cooling_rate)) + 1
    iter = 0

    while T > min_temperature:
        for _ in range(max_iterations):
            i, j = __selection(n)

            current_path[i], current_path[j] = current_path[j], current_path[i]
            temp_dist = __calculate_distance(dmat, current_path)
            
            if not (temp_dist < current_dist or random.random() < e((current_dist - temp_dist) / T)):
                current_path[i], current_path[j] = current_path[j], current_path[i]
                continue

            current_dist = temp_dist
            
            if current_dist < best_dist:
                best_path = current_path.copy()
                best_dist = current_dist
        
        T *= cooling_rate
        iter += 1

        print(f"\r{iter} / {total_iter} ({iter * 100 / total_iter :.4f}%)", end="")

    print()

    return best_path.tolist(), best_dist