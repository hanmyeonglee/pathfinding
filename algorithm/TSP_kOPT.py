import numpy as np

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
    raise NotImplementedError()