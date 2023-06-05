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
from snake_env import *
from settings import settings


class Snakes(Snake):

    def __init__(self, board_size: Tuple[int, int], snake_path: str = " ", snake_name: str = "",
                 number_rounds: int = 100, loading: bool = False,
                 *args, **kwargs):
        super(Snakes, self).__init__(board_size, *args, **kwargs)
        
        if loading:
            self.snake = load_snake(snake_path, snake_name, settings)
        else:
            self.snake = Snake(board_size, *args, **kwargs)
        self.network = self.snake.network
        self.board_size = board_size
        self.hidden_layer_architecture = self.snake.hidden_layer_architecture
        self.hidden_activation = self.snake.hidden_activation
        self.output_activation = self.snake.output_activation
        self.apple_and_self_vision = self.snake.apple_and_self_vision
        self.number_rounds = number_rounds
        self.is_alive = True
        self._fitness = 0
        self.score = 0
        self.lifespan = np.inf
        self._frames = 0

    def calculate_fitness(self):
        
        
        

        pass

    @property
    def fitness(self):
        return self._fitness
        

    def move(self):
        
        for episode in range(self.number_rounds):
            done = False
            obs = self.snake.reset()
            
            while not done:
                
                
                
                obs, reward, done, info = self.snake.step(-1)

            self.snake.calculate_fitness()
            
            
            self.score = (self.score * episode + self.snake.score) / (episode + 1)
            self._fitness = self.score
            self._frames = (self._frames * episode + self.snake._frames) / (episode + 1)

        self.is_alive = False
        

    def update(self):
        pass
        

    def encode_chromosome(self):
        pass

    def decode_chromosome(self):
        pass

    @property
    def chromosome(self):
        pass

    @chromosome.setter
    def chromosome(self, val):
        pass


if __name__ == '__main__':
    env = Snakes(settings["board_size"],
                 "models/test_64",
                        f"snake_1000",
                 number_rounds=10,
                 loading=True
                 )

    while True:
        env.move()
        if env.is_alive:
            break
    print(env.score)