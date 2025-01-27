import matplotlib.pyplot as plt, json

COLORS = "rgmcbykw"

def __draw_points(points: list[tuple], label: int):
    for point in points:
        plt.scatter(*point, marker='o', color=COLORS[label % len(COLORS)], zorder=2)

def __draw_path(points: list[tuple], path: list[int], distance: int | float, i: int):
    path_x, path_y = list(), list()
    for point in path:
        path_x.append(points[point][0])
        path_y.append(points[point][1])
    else:
        path_x.append(points[path[0]][0])
        path_y.append(points[path[0]][1])

    plt.plot(path_x, path_y, linestyle='-', color=COLORS[i % len(COLORS)], label=f"cluster {i + 1} : {distance}", zorder=2)

def __draw_realPath(specific_pathes, path, distance: int | float, i: int):
    path_x, path_y = list(), list()
    for i in range(len(path) - 1):
        start, goal = path[i], path[i + 1]
        for lng, lat in specific_pathes[start][goal]:
            path_x.append(lng)
            path_y.append(lat)
    else:
        for lng, lat in specific_pathes[path[-1]][path[0]]:
            path_x.append(lng)
            path_y.append(lat)

    plt.plot(path_x, path_y, linestyle='-', color=COLORS[i % len(COLORS)], label=f"cluster {i + 1} : {distance}", zorder=2)

def __write_text(points: list[tuple], text: list[str]):
    for (x, y), s in zip(points, text):
        plt.text(x, y * 0.99, s, zorder=2)

def __draw_map():
    map = json.loads(open('./data/map.json', encoding='utf-8').read())

    for road in map['features']:
        coordinates = [list(), list()]
        for x, y in road['geometry']['coordinates']:
            coordinates[0].append(x)
            coordinates[1].append(y)
        
        plt.plot(*coordinates, color='k', zorder=1)

def draw_realPath(
        clusters: list[list[tuple]],
        pathes: list[list[int]],
        distances: list[int | float],
        specific_pathes: list[list],
        texts: list[list[str]] | None = None
):
    n = len(clusters)

    plt.figure()
    plt.axis("equal")
    __draw_map()
    for i in range(n):
        points = [clusters[i][point] for point in pathes[i]]
        __draw_points(points, i)
        __draw_realPath(specific_pathes, pathes[i], distances[i], i)
        if texts is not None:
            __write_text(points, texts[i])

    plt.legend()
    plt.show()

def draw_points_withPath(
        clusters: list[list[tuple]],
        pathes: list[list[int]],
        distances: list[int | float],
        texts: list[list[str]] | None = None
):
    n = len(clusters)

    plt.figure()
    plt.axis("equal")
    for i in range(n):
        points = [clusters[i][point] for point in pathes[i]]
        __draw_points(points, i)
        __draw_path(clusters[i], pathes[i], distances[i], i)
        if texts is not None:
            __write_text(points, texts[i])

    plt.legend()
    plt.show()

def draw_points_withPath_withStartPoint(
        clusters: list[list[tuple]],
        start_point: tuple
    ):
    plt.figure()
    plt.axis("equal")
    for i, cluster in enumerate(clusters):
        for point in cluster:
            plt.scatter(*point, marker='o', color=COLORS[(i + 1) % len(COLORS)])

    plt.scatter(*start_point, marker='o', color=COLORS[0])
    plt.show()

