from math import exp as e, log
from copy import deepcopy

import numpy as np, random

def selection(n: int) -> tuple[int, int]:
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
            i, j = selection(n)

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

    current_path = deepcopy(path)
    current_dist = dist

    best_path = deepcopy(path)
    best_dist = dist

    T = initial_temperature

    total_iter = int(log(min_temperature / initial_temperature, cooling_rate)) + 1
    iter = 0

    while T > min_temperature:
        for _ in range(max_iterations):
            i, j = selection(n)

            left1, middle1, right1 = current_path[i - 1], current_path[i], current_path[(i + 1) % n]
            left2, middle2, right2 = current_path[j - 1], current_path[j], current_path[(j + 1) % n]

            if (j - i == 1) or (j - i == n - 1):
                temp_dist =   current_dist \
                            + (dmat[left1, middle2] + dmat[middle2, middle1] + dmat[middle1, right2]) \
                            - (dmat[left1, middle1] + dmat[middle1, middle2] + dmat[middle2, right2])
            else:
                temp_dist =   current_dist \
                            + (dmat[left1, middle2] + dmat[middle2, right1]) \
                            + (dmat[left2, middle1] + dmat[middle1, right2]) \
                            - (dmat[left1, middle1] + dmat[middle1, right1]) \
                            - (dmat[left2, middle2] + dmat[middle2, right2])
            
            if not (temp_dist < current_dist or random.random() < e((current_dist - temp_dist) / T)):
                continue

            current_path[i], current_path[j] = current_path[j], current_path[i]
            if (a := sum([dmat[current_path[i - 1], current_path[i]].item() for i in range(n)])) != temp_dist:
                print(current_path)
                print(i, j)
                print(current_dist, temp_dist, a)
                print(dmat[left1, middle2] + dmat[middle2, right1])
                print(dmat[left2, middle1] + dmat[middle1, right2])
                print(dmat[left1, middle1] + dmat[middle1, right1])
                print(dmat[left2, middle2] + dmat[middle2, right2])
                raise Exception(f'Error : {current_path} {i} {j}')
            current_dist = temp_dist
            
            if current_dist < best_dist:
                best_path = deepcopy(current_path)
                best_dist = current_dist
        
        T *= cooling_rate
        iter += 1

        print(f"\r{iter} / {total_iter} ({iter * 100 / total_iter :.4f}%)", end="")

    print()

    return best_path, best_dist


def simulated_annealing_insertOp(
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
            i, j = selection(n)

            left1, target, right1 = current_path[i - 1], current_path[i], current_path[(i + 1) % n]

            if abs(i - j) == 1:
                left2, right2 = current_path[j], current_path[(j + 1) % n]
            else:
                left2, right2 = current_path[j - 1], current_path[j]
            
            temp_dist =   current_dist \
                        + (dmat[left1, right1] + dmat[left2, target] + dmat[target, right2]) \
                        - (dmat[left1, target] + dmat[target, right1] + dmat[left2, right2])
            
            if not (temp_dist < current_dist or random.random() < e((current_dist - temp_dist) / T)):
                continue

            current_path = current_path[ : i] + current_path[i + 1 : j + 1] + current_path[i:i+1] + current_path[j + 1 : ]
            current_dist = temp_dist

            if current_dist < best_dist:
                best_path = current_path
                best_dist = current_dist
        
        T *= cooling_rate
        iter += 1

        print(f"\r{iter} / {total_iter} ({iter * 100 / total_iter :.4f}%)", end="")

    print()

    return best_path, best_dist