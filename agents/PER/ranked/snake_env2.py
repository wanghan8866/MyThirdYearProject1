import random
from typing import Tuple

import cv2
import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

from agents.snake_game_gen.patterns import Pattern

SNAKE_LEN_GOAL = 30


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


class SnakeEnv2(gym.Env):

    def __init__(self, size):
        super(SnakeEnv2, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self._frames = 0
        self.fitness = 0
        self.frames_from_last_apple = 0
        self.max_snake_length = int(pow(self.size, 2))

        self.state = np.zeros(shape=(size, size), dtype='uint8')

        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(32,), dtype=np.float64)
        self.pattern = None

        self.apple_num = 0

    def step(self, action):
        self._frames += 1

        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')

        self.state[self.apple_position[0], self.apple_position[1]] = 2

        for position in self.snake_position:
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
            apple_reward = 5

            self.apple_num += 1
            self.frames_from_last_apple = 0

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.frames_from_last_apple += 1
            if self.frames_from_last_apple >= self.max_snake_length:
                self.done = True
                death_reward = -7

        if collision_with_boundaries(self.snake_head, self.size) == 1 or collision_with_self(self.snake_position) == 1:

            if len(self.snake_position) >= self.max_snake_length - 5:
                death_reward = 15
            else:

                death_reward = -10.

            self.done = True

        self.total_reward = len(self.snake_position)

        info = {"apple score": self.apple_num}

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

    def look(self):
        x = self.snake_position[0][0]
        y = self.snake_position[0][1]
        obs = np.array([
            *self.look_at_direction((x, y), (0, -1)),
            *self.look_at_direction((x, y), (1, -1)),
            *self.look_at_direction((x, y), (1, 0)),
            *self.look_at_direction((x, y), (1, 1)),
            *self.look_at_direction((x, y), (0, 1)),
            *self.look_at_direction((x, y), (-1, 1)),
            *self.look_at_direction((x, y), (-1, 0)),
            *self.look_at_direction((x, y), (-1, -1)),
            0., 0., 0., 0., 0., 0., 0., 0.
        ], dtype=np.float64)
        obs[24 + self.direction] = 1
        obs[28 + self.tail_direction] = 1

        return obs

    def look_at_direction(self, current_pos, direction: Tuple[int, int]):
        x = current_pos[0] + direction[0]
        y = current_pos[1] + direction[1]
        distance = abs(direction[0]) + abs(direction[1])
        found_food = 0.
        found_body = 0.
        while not collision_with_boundaries((x, y), self.size):
            if found_body == 0 and self.state[x, y] == 1:
                found_body = 1.
            if found_food == 0 and self.state[x, y] == 2:
                found_food = 1.
            x += direction[0]
            y += direction[1]
            distance += abs(direction[0]) + abs(direction[1])

        return distance / 2. / self.size, found_body, found_food

    def reset(self):

        half = int(self.size // 2)
        self.snake_position = [[half, half], [half - 1, half], [half - 2, half]]
        self.apple_position = [random.randint(1, self.size - 1), random.randint(1, self.size - 1)]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]

        self.apple_num = 0
        self.direction = Direction.RIGHT
        self.frames_from_last_apple = 0
        self.use_pattern(self.pattern)
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.prev_reward = 0
        self._frames = 0
        self.done = False

        observation = np.zeros((32,), dtype=np.float64)
        return observation

    def create_observation(self):
        obs = np.zeros(shape=(10, 10, 4), dtype=np.uint8)
        obs[self.apple_position[0], self.apple_position[1], 3] = 1
        for position in self.snake_position[1:-1]:
            obs[position[0], position[1], 1] = 1
        if not collision_with_boundaries(self.snake_position[0], self.size):
            obs[self.snake_position[0][0], self.snake_position[0][1], 0] = 1
        if not collision_with_boundaries(self.snake_position[-1], self.size):
            obs[self.snake_position[-1][0], self.snake_position[-1][1], 2] = 1

        return obs

    def use_pattern(self, pattern: Pattern):

        if pattern is None:
            return
        self.pattern = pattern
        self.apple_position = [self.pattern.apple.x, self.pattern.apple.y]
        self.snake_position = [[p.x, p.y] for p in self.pattern.snake_body]
        self.snake_head = [self.pattern.snake_body[0].x, self.pattern.snake_body[0].y]

    def render(self, mode='human'):

        size = 25
        self.img = np.zeros((self.size * size, self.size * size, 3), dtype='uint8')
        cv2.rectangle(self.img, (self.apple_position[0] * size, self.apple_position[1] * size),
                      (self.apple_position[0] * size + size, self.apple_position[1] * size + size), (255, 0, 0), -1,
                      lineType=cv2.LINE_AA)

        half = size // 2

        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0] * size, position[1] * size),
                          (position[0] * size + size, position[1] * size + size), (0, 255, 0), 3,
                          lineType=cv2.LINE_AA)
        cv2.rectangle(self.img, (self.snake_position[0][0] * size, self.snake_position[0][1] * size),
                      (self.snake_position[0][0] * size + size, self.snake_position[0][1] * size + size),
                      (200, 0, 125), 3,
                      lineType=cv2.LINE_AA)

        if mode == "human":
            cv2.imshow('a', self.img)

        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv2(size=15)

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

            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    plt.plot(rewards_history)
    plt.plot(avg_reward_history)
    plt.show()
