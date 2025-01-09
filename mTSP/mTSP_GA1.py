import numpy as np

def __one_point_crossover(allocator: np.ndarray, mask: np.ndarray):
    n, v = mask.shape[0], allocator.shape[1]
    crosspoints = np.random.randint(0, v, size=n)
    index_mask = np.tile(np.arange(v), reps=(n, 1)) < crosspoints[:, None]

    return np.where(index_mask, allocator[mask[:, 0]], allocator[mask[:, 1]])


def __swap(generation: np.ndarray, mask: np.ndarray | None = None):
    n, v = generation.shape[0], generation.shape[1]
    if mask is None:
        mask = np.ones(shape=n, dtype=np.bool_)

    rows = np.arange(n)[mask]
    cols01 = np.random.randint(0, v, size=rows.shape[0])
    cols02 = np.random.randint(0, v, size=rows.shape[0])

    generation[rows, cols01], generation[rows, cols02] = \
        generation[rows, cols02], generation[rows, cols01]
    

def __inverse(generation: np.ndarray):
    n, v = generation.shape[0], generation.shape[1]
    
    ij = np.random.randint(0, v, size=(n, 2))
    ij.sort(axis=-1)
    i, j = ij[:, 0], ij[:, 1]

    for x in range(n):
        generation[x, i[x]:j[x]] = np.flip(generation[x, i[x]:j[x]])
    

def __distance(
        k: int,
        start_point: int,
        dmat: np.ndarray,
        generation: np.ndarray,
        allocator: np.ndarray
    ) -> np.ndarray:
    dist = np.zeros(generation.shape[0])
    for i, alloc in enumerate(allocator):
        #print(alloc)
        for j in range(k):
            chromosome = generation[i, alloc == j]
            dist[i] += dmat[start_point, chromosome[0]] + np.sum(dmat[chromosome[:-1], chromosome[1:]]) + dmat[chromosome[-1], start_point]
        
    return dist


def __roulette(distances: np.ndarray) -> np.ndarray:
    mx_distance = np.max(distances)
    mn_distance = np.min(distances)
    return mx_distance + mn_distance - distances


def __selection(roulette: np.ndarray, size: int | tuple) -> np.ndarray:
    return np.random.choice(np.arange(roulette.shape[0]), size=size, p=(roulette / np.sum(roulette)))


def mtsp_genetic_algorithm_allocation(
        k: int,
        dmat: np.ndarray,
        pathes: list[int] | None = None,
        gen_size: int = 64,
        ggap: float = 0.2,
        mutation_probability: float = 0.05,
        threshold: float = 10,
        max_iter: int = 1000
    ) -> tuple[list, int | float]:
    if k == 1:
        raise Exception('Why you use it...?')

    n = dmat.shape[0] - 1
    start_point = 0

    if pathes is None:
        generation = np.tile(np.arange(0, n), reps=(gen_size, 1))
        for chromosome in generation:
            np.random.shuffle(chromosome)
        
        allocator = np.random.randint(0, k, size=(gen_size, n))
    
    else:
        path = []
        alloc = np.zeros(shape=sum(len(p) for p in pathes), dtype=np.uint16)
        prev_ind = 0
        for i, p in enumerate(pathes):
            path += p
            alloc[prev_ind : prev_ind + len(p)] = i
            prev_ind += len(p)
        
        path = np.array(path)
        generation = np.tile(path, reps=(gen_size - 1, 1))
        __swap(generation)
        generation = np.vstack((path, generation))

        allocator = np.tile(alloc, reps=(gen_size - 1, 1))
        __swap(allocator)
        allocator = np.vstack((alloc, allocator))


    n_parents = int(gen_size * ggap)
    n_children = gen_size - n_parents
    
    minimum_distance = np.iinfo(generation.dtype).max
    minimum_path = []
    iter_num = 0

    distances = __distance(k, start_point, dmat, generation, allocator)
    while True:
        survived_parents_indice = np.argsort(distances)[:n_parents]
        survived_parents = generation[survived_parents_indice, :]
        survived_allocator = allocator[survived_parents_indice, :]
        
        
        roulette = __roulette(distances)
        selected_parents_mask = __selection(roulette, size=n_children)
        selected_allocator_mask = __selection(roulette, size=(n_children, 2))


        generation = generation[selected_parents_mask]
        __inverse(generation)
        allocator = __one_point_crossover(allocator, selected_allocator_mask)


        mutated_children_mask = np.random.random(size=n_children) < mutation_probability
        mutated_allocator_mask = np.random.random(size=n_children) < mutation_probability
        __swap(generation, mask=mutated_children_mask)
        __swap(allocator, mask=mutated_allocator_mask)


        generation = np.vstack((survived_parents, generation))
        allocator = np.vstack((survived_allocator, allocator))
        distances = __distance(k, start_point, dmat, generation, allocator)

        current_minimum_distance = np.min(distances)
        if minimum_distance - current_minimum_distance < threshold: iter_num += 1
        else: iter_num = 0

        if current_minimum_distance < minimum_distance:
            minimum_distance = current_minimum_distance
            min_dist_index = np.argmin(distances)
            chromosome = generation[min_dist_index]
            alloc = allocator[min_dist_index]
            minimum_path.clear()
            for i in range(k):
                minimum_path.append(chromosome[alloc == i])


        if iter_num > max_iter:
            break
    
    minimum_distances = []
    for i, route in enumerate(minimum_path):
        minimum_distances.append(
            dmat[start_point, route[0]] + np.sum(dmat[route[:-1], route[1:]]) + dmat[route[-1], start_point]
        )
        minimum_path[i] = route.tolist()
    
    return minimum_path, minimum_distances