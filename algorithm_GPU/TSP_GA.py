import cupy as cp

def __swap(generation: cp.ndarray, mask: cp.ndarray | None = None):
    n, v = generation.shape[0], generation.shape[1]
    if mask is None:
        mask = cp.ones(shape=n, dtype=cp.bool_)

    rows = cp.arange(n)[mask]
    cols01 = cp.random.randint(0, v - 1, size=rows.shape[0])
    cols02 = cp.random.randint(0, v - 1, size=rows.shape[0])

    generation[rows, cols01], generation[rows, cols02] = \
        generation[rows, cols02], generation[rows, cols01]
    

def __inverse(generation: cp.ndarray):
    n, v = generation.shape[0], generation.shape[1]
    index = cp.tile(cp.arange(v), reps=(n, 1))
    
    ij = cp.random.randint(0, v - 1, size=(n, 2))
    ij.sort(axis=-1)
    i, j = ij[:, 0, None], ij[:, 1, None]

    rindex = i + j - index - 1
    mask = cp.logical_and(i <= index, index < j)

    xind = cp.transpose(cp.tile(cp.arange(n), reps=(v, 1)))
    yind = cp.where(mask, rindex, index)
    generation[:, :] = generation[xind, yind]
    

def __distance(dmat: cp.ndarray, generation: cp.ndarray) -> cp.ndarray:
    return cp.sum(
        dmat[generation[:, :-1], generation[:, 1:]], axis=-1
    ) + dmat[generation[:, -1], generation[:, 0]]


def __roulette(distances: cp.ndarray) -> cp.ndarray:
    mx_distance = cp.max(distances)
    mn_distance = cp.min(distances)

    return cp.cumsum(mx_distance + mn_distance - distances)


def __selection(roulette: cp.ndarray, size: int) -> cp.ndarray:
    randnums = cp.random.uniform(0, roulette[-1], size=size)
    mask = cp.sum(roulette <= randnums[:, None], axis=-1)
    return mask


def tsp_genetic_algorithm_gpu(
        dmat: cp.ndarray, path: list[int] | None = None,
        gen_size: int = 1024, ggap: float = 0.2, mutation_probability: float = 0.05,
        threshold: float = 10, max_iter: int = 1000
    ) -> tuple[list, int | float]:
    n = dmat.shape[0]

    if path is None:
        generation = cp.tile(cp.arange(0, n), reps=(gen_size, 1))
        for chromosome in generation:
            cp.random.shuffle(chromosome)
    else:
        path = cp.array(path)
        generation = cp.tile(path, reps=(gen_size - 1, 1))
        __swap(generation)
        generation = cp.vstack((path, generation))

    n_parents = int(gen_size * ggap)
    n_children = gen_size - n_parents
    
    minimum_distance = cp.iinfo(generation.dtype).max
    minimum_path = None
    iter_num = 0

    distances = __distance(dmat, generation)
    while True:
        survived_parents_indice = cp.argsort(distances)[:n_parents]
        survived_parents = generation[survived_parents_indice, :]
        
        selected_parents_mask = __selection(
            __roulette(distances), size=n_children
        )

        generation = generation[selected_parents_mask]
        __swap(generation)
        __inverse(generation)

        mutated_children_mask = cp.random.random(size=n_children) < mutation_probability
        __swap(generation, mask=mutated_children_mask)

        generation = cp.vstack((survived_parents, generation))
        distances = __distance(dmat, generation)

        current_minimum_distance = cp.min(distances)
        if minimum_distance - current_minimum_distance < threshold: iter_num += 1
        else: iter_num = 0

        if current_minimum_distance < minimum_distance:
            minimum_distance = current_minimum_distance
            min_dist_index = cp.argmin(distances)
            minimum_path = generation[min_dist_index].copy()


        if iter_num > max_iter:
            break

    
    return minimum_path.tolist(), minimum_distance