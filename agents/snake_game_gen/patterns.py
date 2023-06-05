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
        
        if node not in path and node.x < 10 and node.x >= 0 and node.y < 10 and node.y >= 0 and graph[
            node.x, node.y] > 0 and graph[node.x, node.y] != 4:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                
                paths.append(newpath)
    return paths


def min_path(graph, start, end):
    paths = find_all_paths(graph, start, end)
    mt = 0
    mpath = []
    
    for path in paths:
        t = len(path)
        
        if t > mt:
            mt = t
            mpath = path
    
    return mpath

    
    
    


class Pattern:
    def __init__(self, frame: np.ndarray):
        head = None
        tail = None
        self.apple = None
        self.snake_body = []
        

        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                if frame[x, y] == 1:
                    head = Point(x, y)
                    
                    
                if frame[x, y] == 3:
                    tail = Point(x, y)
                
                
                if frame[x, y] == 4:
                    self.apple = Point(x, y)

        
        
        self.snake_body = min_path(frame, head, tail)


if __name__ == '__main__':
    
    
    
    
    
    
    
    
    
    
    
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

    
    
    
    
    
    
    
    
    
