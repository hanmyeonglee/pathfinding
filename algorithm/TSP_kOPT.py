import numpy as np, random

def two_opt_sequentialChange(dmat: np.ndarray, path: list[int], dist: int | float) -> tuple[list, int | float]:
    n = len(path)

    improved = True
    while improved:
        improved = False

        for i in range(n - 2):
            for j in range(i + 1, n):
                start1, start2 = path[i - 1], path[j - 1]
                end1, end2 = path[i], path[j]

                changed = (dmat[start1, start2] + dmat[end1, end2]) - \
                            (dmat[start1, end1] + dmat[start2, end2])
                
                if changed < 0:
                    improved = True
                    path = path[ : i] + path[i : j][::-1] + path[j : ]
                    dist += changed
                    break
            
            if improved: break
    
    return path, dist


def two_opt_randomChange(dmat: np.ndarray, path: list[int], dist: int | float) -> tuple[list, int | float]:
    n = len(path)

    indice = np.array([(i, j) for i in range(n - 2) for j in range(i + 1, n)])
    iter_num = 0

    while iter_num < 1:
        iter_num += 1

        improved = True
        while improved:
            improved = False
            np.random.shuffle(indice)

            for x in range((n - 1) * n // 2 - 1):
                i, j = indice[x].tolist()

                start1, start2 = path[i - 1], path[j - 1]
                end1, end2 = path[i], path[j]

                changed = (dmat[start1, start2] + dmat[end1, end2]) - \
                            (dmat[start1, end1] + dmat[start2, end2])
                
                if changed < 0:
                    improved = True
                    path = path[ : i] + path[i : j][::-1] + path[j : ]
                    dist += changed
                    break
    
    return path, dist


def two_opt_FRLS(dmat: np.ndarray, path: list[int], dist: int | float) -> tuple[list, int | float]:
    # first random last sequential
    n = len(path)

    indice = list(range(n - 2))

    improved = True
    while improved:
        improved = False
        random.shuffle(indice)

        for i in indice:
            for j in range(i + 1, n):
                start1, start2 = path[i - 1], path[j - 1]
                end1, end2 = path[i], path[j]

                changed = (dmat[start1, start2] + dmat[end1, end2]) - \
                            (dmat[start1, end1] + dmat[start2, end2])
                
                if changed < 0:
                    improved = True
                    path = path[ : i] + path[i : j][::-1] + path[j : ]
                    dist += changed
                    break
            
            if improved: break
    
    return path, dist


def two_opt_FSLR(dmat: np.ndarray, path: list[int], dist: int | float) -> tuple[list, int | float]:
    # first sequential last random
    n = len(path)

    improved = True
    while improved:
        improved = False

        for i in range(n - 2):
            indice = list(range(i + 1, n))
            random.shuffle(indice)

            for j in indice:
                start1, start2 = path[i - 1], path[j - 1]
                end1, end2 = path[i], path[j]

                changed = (dmat[start1, start2] + dmat[end1, end2]) - \
                            (dmat[start1, end1] + dmat[start2, end2])
                
                if changed < 0:
                    improved = True
                    path = path[ : i] + path[i : j][::-1] + path[j : ]
                    dist += changed
                    break
            
            if improved: break
    
    return path, dist
