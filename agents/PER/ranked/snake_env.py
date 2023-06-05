import math
import random
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

SNAKE_LEN_GOAL = 30


def collision_with_apple(apple_position, score, size):
    apple_position = [random.randint(1, size - 1) * 10, random.randint(1, size - 1) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head, size):
    if snake_head[0] >= size * 10 or snake_head[0] < 0 or snake_head[1] >= size * 10 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]

    if snake_head in snake_position[1:]:

        return 1
    else:
        return 0


class SnakeEnv(gym.Env):

    def __init__(self, size):
        super(SnakeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self.max_snake_length = int(pow(self.size, 2))

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.size * 10, self.size * 10, 3), dtype=np.uint8)
        self.apple_num = 0

    def step(self, action):

        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')

        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)

        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        if action == 0:
            self.snake_head[0] += 10
        elif action == 1:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10
        apple_reward = 0
        death_reward = 0

        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score, self.size)
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
                death_reward = -100. / math.pow(len(self.snake_position), 0.8)

            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position), ord=1)

        self.total_reward = apple_reward + death_reward - euclidean_dist_to_apple / 100.

        info = {"apple score": self.apple_num}

        observation = self.img

        return observation, self.total_reward, self.done, info

    def reset(self):
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')

        half = int(self.size // 2) * 10
        self.snake_position = [[half, half], [half - 10, half], [half - 20, half]]
        self.apple_position = [random.randint(1, self.size - 1) * 10, random.randint(1, self.size - 1) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]

        self.apple_num = 0
        self.direction = 3

        self.prev_reward = 0

        self.done = False

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)

        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = self.img
        return observation

    def render(self, mode='human'):
        if mode == "human":

            cv2.imshow('a', self.img)
        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv(size=15)

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

            print(reward)

            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    plt.plot(rewards_history)
    plt.plot(avg_reward_history)
    plt.show()
