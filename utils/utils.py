import numpy as np

def calculate_distance(dmat: np.ndarray, path: np.ndarray) -> int | float:
    return np.sum(dmat[path[:-1], path[1:]]) + dmat[path[-1], path[0]]