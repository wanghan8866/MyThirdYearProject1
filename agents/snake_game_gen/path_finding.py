import agents.snake_game_gen.snake
from agents.snake_game_gen.misc import *
from agents.snake_game_gen.snake_env3 import Snake
from copy import deepcopy


class BFS:
    def __init__(self, snake: Snake, apple_location: Point):
        self.snake = snake
        self.apple_location = apple_location

    @staticmethod
    def _get_neighbours(node: Point):
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yield Point(node.x + diff[0], node.y + diff[1])

    @staticmethod
    def is_node_in_queue(node: Point, queue: iter):
        """
        Check if element is in a nested list
        """
        return any(node in sublist for sublist in queue)

    def run_bfs(self):
        queue = [[self.snake.snake_array[0].copy()]]

        while queue:
            path = queue[0]
            future_head = path[-1]

            # If snake eats the apple, return the next move after snake's head
            if future_head == self.apple_location:
                return path

            for next_node in self._get_neighbours(future_head):
                if (
                        self.snake.is_invalid_move(next_node)
                        or self.is_node_in_queue(node=next_node, queue=queue)
                ):
                    continue
                new_path = list(path)
                new_path.append(next_node)
                queue.append(new_path)

            queue.pop(0)

    def next_node(self):
        path = self.run_bfs()
        return path[1]


class LongestPath(BFS):
    def __init__(self, snake: Snake, apple_location: Point):

        super().__init__(snake, apple_location)

    def run_longest(self):
        path = self.run_bfs()
        if path is None:
            return

        i = 0
        while True:
            try:
                direction = path[i] - path[i - 1]
            except IndexError:
                break
            snake_path = Snake.create_snake_from_body([10, 10], list(self.snake.snake_array) + path[1:])
            # snake_path.snake_array = snake_path.snake_array + path[1:]
            # print(snake_path.snake_array)

            for neighbour in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if direction == neighbour:
                    x, y = neighbour
                    diff = Point(y, x) if x != 0 else Point(-y, x)
                    extra_node_1 = path[i] + diff
                    extra_node_2 = path[i + 1] + diff
                    if snake_path.is_invalid_move(extra_node_1) or snake_path.is_invalid_move(extra_node_2):
                        i += 1
                    else:
                        path[i + 1:i + 1] = [extra_node_1, extra_node_2]
                    break
            print(*path, sep=", ")
            # print()
            return path[1:]


def heuristic(start: Point, goal: Point):
    return (start.x - goal.x) ** 2 + (start.y - goal.y) ** 2


class Astar(BFS):
    def __init__(self, snake: Snake, apple: Point):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple_location=apple)
        # self.kwargs = kwargs

    def run_astar(self):
        came_from = {}
        close_list = set()
        goal = self.apple_location
        start = self.snake.snake_array[0]
        dummy_snake = Snake.create_snake_from_body([10, 10], list(self.snake.snake_array))
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_list: List[Tuple] = [(fscore[start], start)]
        # print(start, goal, open_list)
        while open_list:
            current = min(open_list, key=lambda x: x[0])[1]
            # print(current)
            open_list.pop(0)
            # print(current)
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                    # print(data)
                return data

            close_list.add(current)

            for neighbor in neighbors:
                neighbor_node = current + neighbor

                if dummy_snake.is_invalid_move(neighbor_node) or neighbor_node in close_list:
                    continue
                if abs(current.x - neighbor_node.x) + abs(current.y - neighbor_node.y) == 2:
                    diff = current - neighbor_node
                    if dummy_snake.is_invalid_move(neighbor_node + (0, diff.y)
                                                   ) or neighbor_node + (0, diff.y) in close_list:
                        continue
                    elif dummy_snake.is_invalid_move(neighbor_node + (diff.x, 0)
                                                     ) or neighbor_node + (diff.x, 0) in close_list:
                        continue
                tentative_gscore = gscore[current] + heuristic(current, neighbor_node)
                if tentative_gscore < gscore.get(neighbor_node, 0) or neighbor_node not in [i[1] for i in open_list]:
                    gscore[neighbor_node] = tentative_gscore
                    fscore[neighbor_node] = tentative_gscore + heuristic(neighbor_node, goal)
                    open_list.append((fscore[neighbor_node], neighbor_node))
                    came_from[neighbor_node] = current


class Mixed:
    def __init__(self, snake, apple_location: Point):
        self.snake = snake
        self.apple_location = apple_location

    def escape(self):
        head = self.snake.snake_array[0]
        largest_neibhour_apple_distance = 0
        newhead = None
        # print("excape")
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbour = head + diff

            if self.snake.is_invalid_move(neighbour):
                continue

            neibhour_apple_distance = (
                    abs(neighbour.x - self.apple_location.x) + abs(neighbour.y - self.apple_location.y)

            )

            # Find the neibhour which has greatest Manhattan distance to apple and has path to tail
            if largest_neibhour_apple_distance < neibhour_apple_distance:
                # snake_tail = None
                snake_tail = self.snake.snake_array[1]
                # Create a virtual snake with a neibhour as head, to see if it has a way to its tail,
                # thus remove two nodes from body: one for moving one step forward, one for avoiding dead checking
                snake = Snake.create_snake_from_body([10, 10], list(self.snake.snake_array)[2:] + [neighbour])
                bfs = BFS(snake=snake, apple_location=snake_tail)
                path = bfs.run_bfs()
                if path is None:
                    continue
                largest_neibhour_apple_distance = neibhour_apple_distance
                newhead = neighbour
        return [head, newhead]

    def run_mixed(self):
        bfs = BFS(self.snake, self.apple_location)
        path = bfs.run_bfs()
        # bfs = Astar(self.snake, self.apple_location)
        # path = bfs.run_astar()
        # if path is None:
        #     return self.escape()
        # else:
        #     return path[-1]

        # agent = LongestPath(self.snake, self.apple_location)
        # path = agent.run_longest()
        # print(self.snake.snake_array[0], "path1", path[-1], )
        if isinstance(path, Point):
            return path
        elif path is None:
            return self.escape()
        else:
            # path = [None] + list(reversed(path))
            if path is None or len(path) == 1:
                # print("here")
                return self.escape()

            # print(self.snake.snake_array[0], "path1", path[1], len(path))
            # print()


            # print("snake head", self.snake.snake_array[0], " new: ", path[0])
            # print()
            length = len(self.snake.snake_array)
            virtual_snake_body = (list(self.snake.snake_array) + path[1:])[-length:]
            virtual_snake_tail = (list(self.snake.snake_array) + path[1:])[-length - 1]
            virtual_snake = Snake.create_snake_from_body([10, 10], virtual_snake_body)
            virtual_snake_longest = BFS(snake=virtual_snake, apple_location=virtual_snake_tail)
            virtual_snake_longest_path = virtual_snake_longest.run_bfs()
            # virtual_snake_longest_path = ""
            if virtual_snake_longest_path is None:
                return self.escape()
            else:
                return path


if __name__ == '__main__':
    l = LongestPath()
    print(Point(0, 1) + Point(20, 1))
