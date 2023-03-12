import math

import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from matplotlib import pyplot as plt

SNAKE_LEN_GOAL = 30


def collision_with_apple(apple_position, score, size):
    apple_position = [random.randint(1, size-1) * 10, random.randint(1, size-1) * 10]
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
        # print(snake_position)
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):

    def __init__(self, size):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self.max_snake_length = int(pow(self.size,2))
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=-500, high=500,
        #                                     shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.size * 10, self.size * 10, 3), dtype=np.uint8)
        self.apple_num = 0

    def step(self, action):
        # self.prev_actions.append(action)
        # cv2.imshow('a', self.img)
        # cv2.waitKey(1)
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # Takes step after fixed time
        # t_end = time.time() + 1
        # k = -1
        # # action = 10
        # while time.time() < t_end:
        #     if k == -1:
        #         # pass
        #         k = cv2.waitKey(1)
        #         # print(k)
        #
        #         if k == 97:
        #             action = 0
        #         elif k == 100:
        #             action = 1
        #         elif k == 119:
        #             action = 3
        #         elif k == 115:
        #             action = 2
        #         # print(k)
        #     else:
        #         continue

        # button_direction = action
        # if self.direction == 0:
        #     if action == 0:
        #         self.direction = 0
        #     elif action == 1:
        #         self.direction = 1
        #     else:
        #         self.direction = 3
        # elif self.direction == 1:
        #     if action == 0:
        #         self.direction = 1
        #     elif action == 1:
        #         self.direction = 2
        #     else:
        #         self.direction = 0
        # elif self.direction == 2:
        #     if action == 0:
        #         self.direction = 2
        #     elif action == 1:
        #         self.direction = 3
        #     else:
        #         self.direction = 1
        # else:
        #     if action == 0:
        #         self.direction = 3
        #     elif action == 1:
        #         self.direction = 0
        #     else:
        #         self.direction = 2
        #
        # # Change the head position based on the button direction
        # if self.direction == 3:
        #     self.snake_head[0] += 10
        # elif self.direction == 1:
        #     self.snake_head[0] -= 10
        # elif self.direction == 2:
        #     self.snake_head[1] += 10
        # elif self.direction == 0:
        #     self.snake_head[1] -= 10

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
        # print(self.snake_head, self.apple_position)
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score, self.size)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 20
            # apple_reward = 250
            self.apple_num += 1

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head, self.size) == 1 or collision_with_self(self.snake_position) == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # self.img = np.zeros((500, 500, 3), dtype='uint8')
            # cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            # cv2.imshow('a', self.img)
            if len(self.snake_position) >= self.max_snake_length - 5:
                death_reward = 150
            else:
                death_reward = -100. / math.pow(len(self.snake_position), 0.8)
            # print("collide", collision_with_boundaries(self.snake_head, self.size),
            #       collision_with_self(self.snake_position))
            # print(self.snake_position)
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position), ord=1)
        # print(euclidean_dist_to_apple)

        # self.total_reward = ((75 - euclidean_dist_to_apple) + apple_reward + death_reward) / 100
        # print(euclidean_dist_to_apple)
        self.total_reward = apple_reward+death_reward - euclidean_dist_to_apple/100.

        # print(self.total_reward)

        # self.reward = self.total_reward - self.prev_reward
        # self.prev_reward = self.total_reward

        # if self.done:
        #     self.reward = -10
        info = {"apple score": self.apple_num}

        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        #
        # snake_length = len(self.snake_position)
        # apple_delta_x = self.apple_position[0] - head_x
        # apple_delta_y = self.apple_position[1] - head_y
        #
        # # create observation:
        #
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # observation = np.array(observation)
        observation = self.img
        # print(observation)
        # print(self.total_reward)
        return observation, self.total_reward, self.done, info

    def reset(self):
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Initial Snake and Apple position
        half = int(self.size // 2) * 10
        self.snake_position = [[half, half], [half - 10, half], [half - 20, half]]
        self.apple_position = [random.randint(1, self.size-1) * 10, random.randint(1, self.size-1) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]
        # print(self.snake_position)
        self.apple_num = 0
        self.direction = 3

        self.prev_reward = 0

        self.done = False

        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        #
        # snake_length = len(self.snake_position)
        # apple_delta_x = self.apple_position[0] - head_x
        # apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be

        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # observation = np.array(observation)
        observation = self.img
        return observation

    def render(self, mode='human'):
        if mode == "human":
            # cv2.waitKey(1)
            cv2.imshow('a', self.img)
        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv(size=15)
    # It will check your custom environment and output additional warnings if needed
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
            # print(f"action: {random_action}")
            obs, reward, done, info = env.step(random_action)
            env.render()
            # A=env.render(mode="rgb_array")
            # print(A.shape)
            print(reward)
            # print(obs.shape)
            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    plt.plot(rewards_history)
    plt.plot(avg_reward_history)
    plt.show()
