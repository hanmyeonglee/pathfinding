import numpy as np

def __average(spliter: np.ndarray, mask: np.ndarray):
    return np.round((spliter[mask[:, 0]] + spliter[mask[:, 1]]) / 2).astype(np.uint16)


def __increase(spliter: np.ndarray, mask: np.ndarray | None = None):
    n, v = spliter.shape
    if mask is None:
        mask = np.ones(shape=n, dtype=np.bool_)

    rows = np.arange(n)[mask]
    cols01 = np.random.randint(0, v, size=rows.shape[0])
    cols02 = np.random.randint(0, v, size=rows.shape[0])

    cols01_mask = spliter[rows, cols01] > 0
    spliter[rows[cols01_mask], cols01[cols01_mask]] -= 1

    cols02_mask = spliter[rows, cols02] < np.sum(spliter[0])
    spliter[rows[cols02_mask], cols02[cols02_mask]] += 1


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
        start_point: int,
        dmat: np.ndarray,
        generation: np.ndarray,
        spliter: np.ndarray
    ) -> np.ndarray:
    indice = np.cumsum(spliter, axis=-1)
    distances = np.zeros(generation.shape[0])
    for i in range(generation.shape[0]):
        chromosome = generation[i, :indice[i, 0]]
        if chromosome.shape[0] != 0: distances[i] += dmat[start_point, chromosome[0]] + np.sum(dmat[chromosome[:-1], chromosome[1:]]) + dmat[chromosome[-1], start_point]

        for j in range(indice.shape[1] - 1):
            chromosome = generation[i, indice[i, j] : indice[i, j+1]]
            if chromosome.shape[0] != 0: distances[i] += dmat[start_point, chromosome[0]] + np.sum(dmat[chromosome[:-1], chromosome[1:]]) + dmat[chromosome[-1], start_point]

    return distances


def __roulette(distances: np.ndarray) -> np.ndarray:
    mx_distance = np.max(distances)
    mn_distance = np.min(distances)
    return mx_distance + mn_distance - distances


def __selection(roulette: np.ndarray, size: int) -> np.ndarray:
    return np.random.choice(np.arange(roulette.shape[0]), size=size, p=(roulette / np.sum(roulette)))


def mtsp_genetic_algorithm_split(
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
        
        split = np.zeros(shape=(gen_size, k), dtype=np.uint16)
        split[:, 0] = np.random.randint(1, n, size=gen_size)
        for i in range(1, k-1):
            split[:, i] = (np.random.rand(gen_size) * (n - np.sum(split, axis=-1))).astype(np.uint16)
        split[:, -1] += n - np.sum(split, axis=-1)

    else:
        path = []
        split = np.zeros(shape=k, dtype=np.uint16)
        for i, p in enumerate(pathes):
            path += p
            split[i] = len(p)
        
        path = np.array(path)
        generation = np.tile(path, reps=(gen_size - 1, 1))
        __swap(generation)
        generation = np.vstack((path, generation))

        spliter = np.tile(split, reps=(gen_size - 1, 1))
        __increase(spliter)
        spliter = np.vstack((split, spliter))
    

    n_parents = int(gen_size * ggap)
    n_children = gen_size - n_parents
    
    minimum_distance = np.iinfo(generation.dtype).max
    minimum_path = []
    iter_num = 0

    distances = __distance(start_point, dmat, generation, spliter)
    while True:
        survived_parents_indice = np.argsort(distances)[:n_parents]
        survived_parents = generation[survived_parents_indice, :]
        survived_spliter = spliter[survived_parents_indice, :]
        

        roulette = __roulette(distances)
        selected_parents_mask = __selection(roulette=roulette, size=n_children)
        selected_spliter_mask = __selection(roulette=roulette, size=(n_children, 2))


        generation = generation[selected_parents_mask]
        __inverse(generation)
        spliter = __average(spliter=spliter, mask=selected_spliter_mask)


        mutated_children_mask = np.random.random(size=n_children) < mutation_probability
        mutated_spliter_mask = np.random.random(size=n_children) < mutation_probability
        __swap(generation, mask=mutated_children_mask)
        __increase(spliter, mask=mutated_spliter_mask)


        generation = np.vstack((survived_parents, generation))
        spliter = np.vstack((survived_spliter, spliter))
        distances = __distance(start_point, dmat, generation, spliter)

        current_minimum_distance = np.min(distances)
        if minimum_distance - current_minimum_distance < threshold: iter_num += 1
        else: iter_num = 0

        if current_minimum_distance < minimum_distance:
            minimum_distance = current_minimum_distance
            min_dist_index = np.argmin(distances)
            chromosome = generation[min_dist_index]
            index = np.cumsum(spliter[min_dist_index])
            minimum_path.clear()
            minimum_path.append(chromosome[:index[0]])
            for i in range(index.shape[0] - 1):
                minimum_path.append(chromosome[index[i]:index[i+1]])


        if iter_num > max_iter:
            break


    minimum_distances = []
    for i, route in enumerate(minimum_path):
        minimum_distances.append(
            dmat[start_point, route[0]] + np.sum(dmat[route[:-1], route[1:]]) + dmat[route[-1], start_point]
        )
        minimum_path[i] = route.tolist()
    
    return minimum_path, minimum_distances