import matplotlib.pyplot as plt

COLORS = "rgmcbykw"

def draw_points(points: list[tuple], path: list[int], label: int):
    points = [points[point] for point in path]
    for point in points:
        plt.scatter(*point, marker='o', color=COLORS[label])

def draw_path(points: list[tuple], path: list[int], distance: int | float, i: int):
    path_x, path_y = list(), list()
    for point in path:
        path_x.append(points[point][0])
        path_y.append(points[point][1])
    else:
        path_x.append(points[path[0]][0])
        path_y.append(points[path[0]][1])

    plt.plot(path_x, path_y, linestyle='-', color=COLORS[i], label=f"cluster {i + 1} : {distance}")

def draw_points_withPath(clusters: list[list[tuple]], pathes: list[list[int]], distances: list[int | float]):
    n = len(clusters)

    plt.figure()
    plt.axis("equal")
    for i in range(n):
        draw_points(clusters[i], pathes[i], i)
        draw_path(clusters[i], pathes[i], distances[i], i)

    plt.legend()
    plt.show()