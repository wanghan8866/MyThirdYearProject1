import math
import random
from typing import Tuple

import cv2
import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

angle_16 = math.tan(22.5 / 180 * math.pi)


def plotPixel(x1, y1, x2, y2, dx, dy, decide):
    pk = 2 * dy - dx
    points = []

    for i in range(0, dx + 1):
        if decide:

            points.append((y1, x1))

        else:

            points.append((x1, y1))

        if (x1 < x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):

            if (decide == 0):

                pk = pk + 2 * dy
            else:

                pk = pk + 2 * dy
        else:
            if (y1 < y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            pk = pk + 2 * dy - 2 * dx
    return points


def findPoints(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if (dx > dy):

        points = plotPixel(x1, y1, x2, y2, dx, dy, 0)


    else:

        points = plotPixel(y1, x1, y2, x2, dy, dx, 1)

    return points


class Direction:
    UP = 0
    RIGHT = 3
    DOWN = 1
    LEFT = 2

    @staticmethod
    def toString(dir):
        if dir == 0:
            return "UP"
        elif dir == 3:
            return "RIGHT"
        elif dir == 1:
            return "DOWN"
        elif dir == 2:
            return "LEFT"


def collision_with_apple(score, size, current_state):
    apple_position = [random.randint(1, size - 1), random.randint(1, size - 1)]
    score += 1
    try:
        while current_state[apple_position[0], apple_position[1]]:
            apple_position = [random.randint(1, size - 1), random.randint(1, size - 1)]
    except Exception as e:
        print(apple_position)
        raise e
    return apple_position, score


def collision_with_boundaries(snake_head, size):
    if snake_head[0] >= size or snake_head[0] < 0 or snake_head[1] >= size or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]

    if snake_head in snake_position[1:]:

        return 1
    else:
        return 0


class SnakeEnv3(gym.Env):

    def __init__(self, size):
        super(SnakeEnv3, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self.max_snake_length = int(pow(self.size, 2))

        self.state = np.zeros(shape=(size, size), dtype='uint8')

        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(56,), dtype=np.float64)

        self.apple_num = 0
        self.max_index = self.size - 1

    def step(self, action):

        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')

        cv2.rectangle(self.img, (self.apple_position[0] * 10, self.apple_position[1] * 10),
                      (self.apple_position[0] * 10 + 10, self.apple_position[1] * 10 + 10), (0, 0, 255), 3)
        self.state[self.apple_position[0], self.apple_position[1]] = 2

        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0] * 10, position[1] * 10),
                          (position[0] * 10 + 10, position[1] * 10 + 10), (0, 255, 0), 3)
            self.state[position[0], position[1]] = 1

        if action == -1:
            action = self.direction

        if action == Direction.RIGHT:
            self.snake_head[0] += 1
            self.direction = Direction.RIGHT
        elif action == Direction.LEFT:
            self.snake_head[0] -= 1
            self.direction = Direction.LEFT
        elif action == Direction.DOWN:
            self.snake_head[1] += 1
            self.direction = Direction.DOWN
        elif action == Direction.UP:
            self.snake_head[1] -= 1
            self.direction = Direction.UP

        apple_reward = 0
        death_reward = 0

        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.score, self.size, current_state=self.state)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 20

            self.apple_num += 1

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        if collision_with_boundaries(self.snake_head, self.size) == 1 or collision_with_self(self.snake_position) == 1:

            if len(self.snake_position) >= self.max_snake_length - 5:
                death_reward = 150
            else:

                death_reward = -100.

            self.done = True

        self.total_reward = apple_reward + death_reward

        info = {"apple score": self.apple_num}

        observation = self.img

        if self.snake_position[-2][0] + 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.LEFT
        elif self.snake_position[-2][0] - 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.RIGHT
        elif self.snake_position[-2][1] + 1 == self.snake_position[-1][1]:
            self.tail_direction = Direction.UP
        else:
            self.tail_direction = Direction.DOWN

        self.observation = self.look()

        return self.observation, self.total_reward, self.done, info

    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.snake_array[0].copy()
        distance = 1.0
        total_distance = 0.0

        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False
        food_found = False

        while self._within_wall(position):
            if not body_found and self._is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
                self.intersections.append(self_location)
            if not food_found and self._is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                self.intersections.append(apple_location)
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        assert (total_distance != 0.0)

        dist_to_wall = 1.0 / total_distance

        if self.apple_and_self_vision == 'binary':
            dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
            dist_to_self = 1.0 if dist_to_self != np.inf else 0.0

        elif self.apple_and_self_vision == 'distance':
            dist_to_apple = 1.0 / dist_to_apple
            dist_to_self = 1.0 / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)

        self.intersections.append(wall_location)
        return (vision, drawable_vision)

    def look(self):
        x = self.snake_position[0][0]
        y = self.snake_position[0][1]
        dy1 = y * angle_16
        dy1 = 0 if dy1 < 1e-3 else dy1
        dy2 = abs(self.max_index - y) * angle_16
        dy2 = 0 if dy2 < 1e-3 else dy2
        dx1 = x * angle_16
        dx1 = 0 if dx1 < 1e-3 else dx1
        dx2 = abs(self.max_index - x) * angle_16
        dx2 = 0 if dx2 < 1e-3 else dx2

        obs = np.array([
            *self.look_at_direction((x, y), (0, -1)),
            *self.look_at_direction_16((x, y), (x + dy1, 0)),
            *self.look_at_direction((x, y), (1, -1)),
            *self.look_at_direction_16((x, y), (self.max_index, y - dx2)),
            *self.look_at_direction((x, y), (1, 0)),
            *self.look_at_direction_16((x, y), (self.max_index, y + dx2)),
            *self.look_at_direction((x, y), (1, 1)),
            *self.look_at_direction_16((x, y), (x + dy2, self.max_index)),

            *self.look_at_direction((x, y), (0, 1)),
            *self.look_at_direction_16((x, y), (x - dy2, self.max_index)),
            *self.look_at_direction((x, y), (-1, 1)),
            *self.look_at_direction_16((x, y), (0, y + dx1)),

            *self.look_at_direction((x, y), (-1, 0)),
            *self.look_at_direction_16((x, y), (0, y - dx1)),
            *self.look_at_direction((x, y), (-1, -1)),
            *self.look_at_direction_16((x, y), (x - dy1, 0)),
            0., 0., 0., 0., 0., 0., 0., 0.
        ], dtype=np.float64)
        obs[48 + self.direction] = 1
        obs[52 + self.tail_direction] = 1

        return obs

    def look_at_direction(self, current_pos, direction: Tuple[int, int]):
        print("d", direction)
        x = current_pos[0] + direction[0]
        y = current_pos[1] + direction[1]
        distance = abs(direction[0]) + abs(direction[1])
        found_food = 0
        found_body = 0
        while not collision_with_boundaries((x, y), self.size):
            if found_body == 0 and self.state[x, y] == 1:
                found_body = 1.
            if found_food == 0 and self.state[x, y] == 2:
                found_food = 1.
            x += direction[0]
            y += direction[1]
            distance += abs(direction[0]) + abs(direction[1])

        cv2.line(self.img, (current_pos[0] * 10 + 5, current_pos[1] * 10 + 5), (x * 10 + 5, y * 10 + 5),
                 (255, 100, 100), thickness=1)
        return distance / 2. / self.size, found_body, found_food

    def look_at_direction_16(self, current_pos, new_pos: Tuple[float, float]):
        print(new_pos)
        x = current_pos[0]
        y = current_pos[1]
        x2 = new_pos[0]
        y2 = new_pos[1]
        points = findPoints(x, y, x2, y2)
        found_food = 0
        found_body = 0

        for point in points[1:]:
            if collision_with_boundaries(point, self.size):
                x2 = point[0]
                y2 = point[1]
                break
            x2 = point[0]
            y2 = point[1]

            if found_body == 0 and self.state[x2, y2] == 1:
                found_body = 1.
            if found_food == 0 and self.state[x2, y2] == 2:
                found_food = 1.

        distance = abs(x2 - x) + abs(y2 - y)

        cv2.line(self.img, (x * 10 + 5, y * 10 + 5), (x2 * 10 + 5, y2 * 10 + 5),
                 (255, 100, 100), thickness=1)
        return (distance + 1) / 2. / self.size, found_body, found_food

    def reset(self):
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')

        half = int(self.size // 2)
        self.snake_position = [[half, half], [half - 1, half], [half - 2, half]]
        self.apple_position = [random.randint(1, self.size - 1), random.randint(1, self.size - 1)]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]

        self.apple_num = 0
        self.direction = Direction.RIGHT
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.prev_reward = 0

        self.done = False

        observation = np.zeros((56,), dtype=np.float64)
        return observation

    def render(self, mode='human'):
        if mode == "human":

            cv2.imshow('a', self.img)


        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv3(size=15)

    check_env(env)
    episodes = 20
    rewards_history = []
    avg_reward_history = []
    print(env.observation_space.shape)
    for episode in range(episodes):
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            random_action = env.action_space.sample()

            obs, reward, done, info = env.step(random_action)
            env.render()

            print("reward", reward)
            for i in range(0, 16):
                print(i, ", {:.4f}, {}, {}".format(*obs[3 * (i):3 * (i + 1)]))
            print(obs[-4:])
            print(obs[-8:-4])
            print()

            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    plt.show()
