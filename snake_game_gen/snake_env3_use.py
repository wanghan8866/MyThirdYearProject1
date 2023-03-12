import gym
from gym import Env
import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json
from gym import spaces
import cv2
import time
from misc import *
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, \
    get_activation_by_name
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from Win_counter import WinCounter
from patterns import Pattern
from path_finding import Mixed
from snake_env3 import *

from settings import settings

PRINT_NUM = 1
if __name__ == '__main__':
    displaying = True
    using_path_finding = False
    path_correctness = []
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
    # env = Snake.create_snake_from_pattern((10,10), p)
    #
    # env = Snake([10, 10], hidden_layer_architecture=settings['hidden_network_architecture'],
    #             hidden_activation=settings['hidden_layer_activation'],
    #             output_activation=settings['output_layer_activation'],
    #             lifespan=settings['lifespan'],
    #             apple_and_self_vision=settings['apple_and_self_vision'])
    X = np.array(
        [200, 400, 600, 800, 1000, 1200, 1400, 1450, 1500, 1550, 1575, 1600, 1625, 1650, 1675, 1690, 1695, 1699])
    X = [400]
    y_scores = []
    y_steps = []
    for name in X:
        env = load_snake("models/test_64", f"snake_{name}", settings)
        # env.use_pattern(p)
        # It will check your custom environment and output additional warnings if needed
        # check_env(env)
        episodes = 100
        rewards_history = []
        avg_reward_history = []
        path_history = []
        # print(env.observation_space.shape)

        for episode in range(episodes):
            path_correctness.append([])
            done = False
            obs = env.reset()
            rewards = 0
            while not done:
                action = env.possible_directions[env.action_space.sample()]
                action = -1
                path = None

                path = Mixed(env, env.apple_location).run_mixed()
                # print("path: ", path)
                if path is None:
                    action = -1
                    # print("path: ", path)
                    # print("path: ", path)
                elif path[1] is None:
                    action = -1
                else:
                    # print("path: ", *path)
                    if using_path_finding:
                        result = path[1] - env.snake_array[0]
                        old_action = action
                        if result == Point(0, 1):
                            action = "d"
                        elif result == Point(0, -1):
                            action = "u"
                        elif result == Point(1, 0):
                            action = "r"
                        else:
                            action = "l"
                        if old_action == action:
                            path_correctness[episode].append(1)
                        else:
                            path_correctness[episode].append(0)
                    else:
                        action = -1
                    # print(old_action, action)

                if displaying:
                    t_end = time.time() + 0.1
                    k = -1
                    # action = -1
                    while time.time() < t_end:
                        if k == -1:
                            # pass
                            k = cv2.waitKey(1)
                            # print(k)

                            if k == 97:
                                action = "l"
                            elif k == 100:
                                action = "r"
                            elif k == 119:
                                action = "u"
                            elif k == 115:
                                action = "d"
                            # print(k)
                        else:
                            continue

                    env.render(drawing_vision=False, path=path)

                # action=-1

                # print(action)
                obs, reward, done, info = env.step(action)
                # print("reward", reward)

                # A=env.render(mode="rgb_array")
                # print(A.shape)
                # print("reward", reward)
                # print(obs.shape)
                # rewards += reward
            # rewards_history.append(rewards)
            avg_reward = len(env.snake_array)
            avg_reward_history.append(avg_reward)
            if len(env.steps_history) == 0:
                path_history.append(0)
            else:
                path_history.append(np.mean(env.steps_history))
            # if episode % PRINT_NUM == 0: print(f"games: {episode + 1}, avg_score: {avg_reward}, path: {np.mean(
            # path_correctness[episode])}, steps: {env._frames}, step history:{env.steps_history}")
        print()
        print(f"snake: snake_{name}")
        print(f"games: average reward over {episodes} games: {np.mean(avg_reward_history)}")
        print(f"games: std reward over {episodes} games: {np.std(avg_reward_history)}")
        print(f"games: average steps over {episodes} games: {np.mean(path_history)}")
        print(f"games: std steps over {episodes} games: {np.std(path_history)}")
        y_scores.append(np.mean(avg_reward_history))
        y_steps.append(np.mean(path_history))
    plt.plot(X, y_scores)
    plt.show()
    plt.plot(X, y_steps)
    plt.show()
