import math

import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from matplotlib import pyplot as plt
from typing import Tuple
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
        # print(snake_position)
        return 1
    else:
        return 0


class SnakeEnv2(gym.Env):

    def __init__(self, size):
        super(SnakeEnv2, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self._frames = 0
        self.fitness = 0
        self.frames_from_last_apple = 0
        self.max_snake_length = int(pow(self.size, 2))
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=-500, high=500,
        #                                     shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float64)
        self.state = np.zeros(shape=(size, size), dtype='uint8')
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(self.size * 10, self.size * 10, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(32,), dtype=np.float64)
        self.pattern = None

        self.apple_num = 0

    def step(self, action):
        self._frames += 1
        # self.prev_actions.append(action)
        # cv2.imshow('a', self.img)
        # cv2.waitKey(1)
        # print("action in DQN",action)
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        # self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Display Apple
        # cv2.rectangle(self.img, (self.apple_position[0] * 10, self.apple_position[1] * 10),
        #               (self.apple_position[0] * 10 + 10, self.apple_position[1] * 10 + 10), (0, 0, 255), 3)
        self.state[self.apple_position[0], self.apple_position[1]] = 2
        # Display Snake
        for position in self.snake_position:
            # cv2.rectangle(self.img, (position[0] * 10, position[1] * 10),
            #               (position[0] * 10 + 10, position[1] * 10 + 10), (0, 255, 0), 3)
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
        # print(self.snake_head, self.apple_position)
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.score, self.size, current_state=self.state)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 5
            # apple_reward = 250
            self.apple_num += 1
            self.frames_from_last_apple = 0

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.frames_from_last_apple += 1
            if self.frames_from_last_apple >= self.max_snake_length:
                self.done = True
                death_reward = -7

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head, self.size) == 1 or collision_with_self(self.snake_position) == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # self.img = np.zeros((500, 500, 3), dtype='uint8')
            # cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            # cv2.imshow('a', self.img)
            if len(self.snake_position) >= self.max_snake_length - 5:
                death_reward = 15
            else:
                # death_reward = -100. / math.pow(len(self.snake_position), 0.8)
                death_reward = -10.
            # print("collide", collision_with_boundaries(self.snake_head, self.size),
            #       collision_with_self(self.snake_position))
            # print(self.snake_position)
            self.done = True

        # euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position), ord=1)
        # print(euclidean_dist_to_apple)

        # self.total_reward = ((75 - euclidean_dist_to_apple) + apple_reward + death_reward) / 100
        # print(euclidean_dist_to_apple)
        # self.total_reward = apple_reward + death_reward
        self.total_reward = len(self.snake_position)

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
        # observation = self.img
        # print(observation)
        # print(self.total_reward)
        # print(self.state.T)
        if self.snake_position[-2][0] + 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.LEFT
        elif self.snake_position[-2][0] - 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.RIGHT
        elif self.snake_position[-2][1] + 1 == self.snake_position[-1][1]:
            self.tail_direction = Direction.UP
        else:
            self.tail_direction = Direction.DOWN
        # print("head:", Direction.toString(self.direction), Direction.toString(self.tail_direction))
        self.observation = self.look()
        # print("obs", observation)
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
        # up, upright,right,rightdown, down, downleft,left,leftup
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
        # print((x, y), found_body, found_food, distance)
        # cv2.line(self.img, (current_pos[0] * 10 + 5, current_pos[1] * 10 + 5), (x * 10 + 5, y * 10 + 5),
        #          (255, 100, 100), thickness=1)
        return distance / 2. / self.size, found_body, found_food

    def reset(self):
        # self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Initial Snake and Apple position
        half = int(self.size // 2)
        self.snake_position = [[half, half], [half - 1, half], [half - 2, half]]
        self.apple_position = [random.randint(1, self.size - 1), random.randint(1, self.size - 1)]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]
        # print(self.snake_position)
        self.apple_num = 0
        self.direction = Direction.RIGHT
        self.frames_from_last_apple = 0
        self.use_pattern(self.pattern)
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.prev_reward = 0
        self._frames = 0
        self.done = False

        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        #
        # snake_length = len(self.snake_position)
        # apple_delta_x = self.apple_position[0] - head_x
        # apple_delta_y = self.apple_position[1] - head_y
        #
        # self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        #
        # for i in range(SNAKE_LEN_GOAL):
        #     self.prev_actions.append(-1)  # to create history

        # create observation:
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # observation = np.array(observation)
        # observation = self.img
        observation = np.zeros((32,), dtype=np.float64)
        return observation

    def create_observation(self):
        obs = np.zeros(shape=(10, 10, 4), dtype=np.uint8)
        obs[self.apple_position[0], self.apple_position[1], 3] = 1
        for position in self.snake_position[1:-1]:
            # cv2.rectangle(self.img, (position[0] * 10, position[1] * 10),
            #               (position[0] * 10 + 10, position[1] * 10 + 10), (0, 255, 0), 3)
            obs[position[0], position[1], 1] = 1
        if not collision_with_boundaries(self.snake_position[0], self.size):
            obs[self.snake_position[0][0], self.snake_position[0][1], 0] = 1
        if not collision_with_boundaries(self.snake_position[-1], self.size):
            obs[self.snake_position[-1][0], self.snake_position[-1][1], 2] = 1

        return obs

    def use_pattern(self, pattern: Pattern):
        # print("use pattern in Q")
        if pattern is None:
            return
        self.pattern = pattern
        self.apple_position = [self.pattern.apple.x, self.pattern.apple.y]
        self.snake_position = [[p.x, p.y] for p in self.pattern.snake_body]
        self.snake_head = [self.pattern.snake_body[0].x, self.pattern.snake_body[0].y]
        # self._body_locations = set(self.pattern.snake_body)

    def render(self, mode='human'):

        # obs = self.create_observation()
        # for row in obs:
        #     for col in row:
        #         value = col
        #         # if col[0] == 1:
        #         #     value = 1
        #         # elif col[1] == 1:
        #         #     value = 2
        #         # elif col[2] == 1:
        #         #     value = 3
        #         # elif col[3] == 1:
        #         #     value = 4
        #         print(value, end=", ")
        #     print()
        size = 25
        self.img = np.zeros((self.size * size, self.size * size, 3), dtype='uint8')
        cv2.rectangle(self.img, (self.apple_position[0] * size, self.apple_position[1] * size),
                      (self.apple_position[0] * size + size, self.apple_position[1] * size + size), (255, 0, 0), -1,
                      lineType=cv2.LINE_AA)

        # Display Snake
        # print([ str(p) for p in self.snake_array])
        half = size // 2
        # for vision in self._drawable_vision:
        #     # print(vision.wall_location)
        #     if vision.wall_location is None:
        #         continue
        #     cv2.line(self.img, (self.snake_array[0].x * size + half, self.snake_array[0].y * size + half),
        #              (vision.wall_location.x * size + half, vision.wall_location.y * size + half),
        #              (255, 100, 100), thickness=1, lineType=cv2.LINE_AA)
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0] * size, position[1] * size),
                          (position[0] * size + size, position[1] * size + size), (0, 255, 0), 3,
                          lineType=cv2.LINE_AA)
        cv2.rectangle(self.img, (self.snake_position[0][0] * size, self.snake_position[0][1] * size),
                      (self.snake_position[0][0] * size + size, self.snake_position[0][1] * size + size),
                      (200, 0, 125), 3,
                      lineType=cv2.LINE_AA)
        # cv2.waitKey(1)
        if mode == "human":
            cv2.imshow('a', self.img)
            # print(self.observation)
        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv2(size=15)
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
            print("reward", reward)
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
