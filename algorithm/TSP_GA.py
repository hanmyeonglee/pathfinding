from __future__ import annotations
import numpy as np

def __swap(generation: np.ndarray, mask: np.ndarray | None = None):
    n, v = generation.shape[0], generation.shape[1]
    if mask is None:
        mask = np.ones(shape=n, dtype=np.bool_)

    rows = np.arange(n)[mask]
    cols01 = np.random.randint(0, v - 1, size=rows.shape[0])
    cols02 = np.random.randint(0, v - 1, size=rows.shape[0])

    generation[rows, cols01], generation[rows, cols02] = \
        generation[rows, cols02], generation[rows, cols01]
    

def __distance(dmat: np.ndarray, generation: np.ndarray) -> np.ndarray:
    return np.sum(
        dmat[generation[:, :-1], generation[:, 1:]], axis=-1
    ) + dmat[generation[:, -1], generation[:, 0]]


def __roulette(distances: np.ndarray) -> np.ndarray:
    mx_distance = np.max(distances)
    mn_distance = np.min(distances)

    return np.cumsum(mx_distance + mn_distance - distances)


def __selection(roulette: np.ndarray, size: int) -> np.ndarray:
    randnums = np.random.uniform(0, roulette[-1], size=size)
    mask = np.sum(roulette <= randnums[:, None], axis=-1)
    return mask


def tsp_genetic_algorithm(
        dmat: np.ndarray, path: list[int] | None = None,
        gen_size: int = 64, ggap: float = 0.2, mutation_probability: float = 0.05,
        threshold: float = 10, max_iter: int = 1000
    ) -> tuple[list, int | float]:
    n = dmat.shape[0]

    if path is None:
        generation = np.tile(np.arange(0, n), reps=(gen_size, 1))
        for chromosome in generation:
            np.random.shuffle(chromosome)
    else:
        path = np.array(path)
        generation = np.tile(path, reps=(gen_size - 1, 1))
        __swap(generation)
        generation = np.vstack((path, generation))

    n_parents = int(gen_size * ggap)
    n_children = gen_size - n_parents
    
    minimum_distance = np.iinfo(generation.dtype).max
    minimum_path = None
    iter_num = 0

    distances = __distance(dmat, generation)
    while True:
        survived_parents_indice = np.argsort(distances)[:n_parents]
        survived_parents = generation[survived_parents_indice, :]
        
        roulette = __roulette(distances)
        selected_parents_mask = __selection(roulette, size=n_children)

        generation = generation[selected_parents_mask]
        __swap(generation)

        mutated_children_mask = np.random.random(size=n_children) < mutation_probability
        __swap(generation, mask=mutated_children_mask)

        generation = np.vstack((survived_parents, generation))
        distances = __distance(dmat, generation)

        current_minimum_distance = np.min(distances)
        if minimum_distance - current_minimum_distance < threshold: iter_num += 1
        else: iter_num = 0

        if current_minimum_distance < minimum_distance:
            minimum_distance = current_minimum_distance
            min_dist_index = np.argmin(distances)
            minimum_path = generation[min_dist_index].copy()


        if iter_num > max_iter:
            break

    
    return minimum_path.tolist(), minimum_distance