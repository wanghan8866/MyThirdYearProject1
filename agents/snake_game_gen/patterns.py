from typing import List, Tuple
from agents.snake_game_gen.misc import Point
import numpy as np


def find_all_paths(graph, start, end, path=None):
    path = [] if path is None else path
    path = path + [start]
    if start.x == end.x and start.y == end.y:
        return [path]
    if start.x > 10 or start.x < 0 or start.y > 10 or start.y < 0:
        return []
    paths = []
    for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        node = start + diff
        # print(node)
        if node not in path and node.x < 10 and node.x >= 0 and node.y < 10 and node.y >= 0 and graph[
            node.x, node.y] > 0 and graph[node.x, node.y] != 4:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                # print(*newpath)
                paths.append(newpath)
    return paths


def min_path(graph, start, end):
    paths = find_all_paths(graph, start, end)
    mt = 0
    mpath = []
    # print('\tAll paths:', )
    for path in paths:
        t = len(path)
        # print('\t\tevaluating:', *path, t)
        if t > mt:
            mt = t
            mpath = path
    # print("best: ", *mpath)
    return mpath

    # e1 = ' '.join('{}->{}:{}'.format(i, j, graph[i][j]) for i, j in zip(mpath, mpath[1::]))
    # e2 = str(sum(graph[i][j] for i, j in zip(mpath, mpath[1::])))
    # print('Best path: ' + e1 + '   Total: ' + e2 + '\n')


class Pattern:
    def __init__(self, frame: np.ndarray):
        head = None
        tail = None
        self.apple = None
        self.snake_body = []
        # print("Pattern:",frame)

        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                if frame[x, y] == 1:
                    head = Point(x, y)
                    # self.snake_body.append(head)
                    # print(head)
                if frame[x, y] == 3:
                    tail = Point(x, y)
                # if frame[x, y] > 0 and frame[x, y] != 4:
                #     self.snake_body.append(Point(x, y))
                if frame[x, y] == 4:
                    self.apple = Point(x, y)

        # self.snake_body.sort(key=lambda p: abs(head.x - p.x) + abs(head.y - p.y))
        # print(*self.snake_body)
        self.snake_body = min_path(frame, head, tail)


if __name__ == '__main__':
    # p=Pattern(np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    p = Pattern(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    p = Pattern(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 1, 0, 0, 4, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    p = Pattern(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 2, 3, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 2, 0, 0, 0, 0],
        [2, 2, 2, 0, 4, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    # graph = {'D1': {'D2': 1, 'C1': 1},
    #          'D2': {'C2': 1, 'D1': 1},
    #          'C1': {'C2': 1, 'B1': 1, 'D1': 1},
    #          'C2': {'D2': 1, 'C1': 1, 'B2': 1},
    #          'B1': {'C1': 1, 'B2': 1},
    #          'B2': {'B1': 1, 'A2': 1, 'C2': 1},
    #          'A2': {'B2': 1, 'A1': 1},
    #          'A1': {'A2': 1}}
    # min_path(graph, 'D1', 'A1')
