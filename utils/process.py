import json, numpy as np

def make_distanceMatrix(
        datafile_name: str = './data/dmat.json',
        encoding: str = 'utf-8',
        condition: str = 'distance'
) -> tuple[np.ndarray, list, list[list]]:
    assert condition in ('distance', 'time', 'cost'), "condition is wrong."

    data = json.loads(open(datafile_name, encoding=encoding).read())
    n = len(data)
    
    dmat = np.zeros((n, n))
    points = [None for _ in range(n)]
    path = [[None for _ in range(n)] for _ in range(n)]
    
    match condition:
        case 'distance':
            keys = ('distance', )
        case 'time':
            keys = ('duration', )
        case 'cost':
            keys = ('fuelPrice', 'tollFare', )
    
    for i in range(n):
        for j in range(n):
            if i == j:
                points[i] = data[i][j]["start"]

            dmat[i, j] = sum(data[i][j][key] for key in keys)
            path[i][j] = data[i][j]["path"]

    return dmat, points, path