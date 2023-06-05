import csv
import math
import os
import random
from time import time
from typing import List, Tuple, Union, Optional

import numpy as np

from agents.snake_game_gen.Win_counter import WinCounter
from agents.snake_game_gen.genetic_algorithm.crossover import simulated_binary_crossover as SBX
from agents.snake_game_gen.genetic_algorithm.crossover import single_point_binary_crossover
from agents.snake_game_gen.genetic_algorithm.individual import Individual
from agents.snake_game_gen.genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from agents.snake_game_gen.genetic_algorithm.population import Population
from agents.snake_game_gen.genetic_algorithm.selection import ellitism_selection, roulette_wheel_selection
from agents.snake_game_gen.settings import settings as GLOBAL_SETTINGS
from agents.snake_game_gen.snake_env3 import Snake, save_snake
from agents.snake_game_gen.tk_nn import NN_canvas
from training.base_training_env import BaseTrainingEnv


def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)


def save_stats(population: Population, path_to_dir: str, fname: str):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    f = os.path.join(path_to_dir, fname + '.csv')

    frames = [individual._frames for individual in population.individuals]
    apples = [individual.score for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]

    write_header = True
    if os.path.exists(f):
        write_header = False

    trackers = [('steps', frames),
                ('apples', apples),
                ('fitness', fitness)
                ]
    stats = ['mean', 'median', 'std', 'min', 'max']

    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(f, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}

        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        writer.writerow(row)


def load_stats(path_to_stats: str, normalize: Optional[bool] = True):
    data = {}

    fieldnames = None
    trackers_stats = None
    trackers = None
    stats_names = None

    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)

        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}

            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)

    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data


class GeneticSnakeTrainingEnv(BaseTrainingEnv):
    def __init__(self, setting: dict, path: str, times: int, current_gen: int = 0, create_snake=None, *args, **kwargs):
        super(GeneticSnakeTrainingEnv, self).__init__(setting, *args, **kwargs)
        self._SBX_eta = GLOBAL_SETTINGS['SBX_eta']
        self._mutation_bins = np.cumsum([GLOBAL_SETTINGS['probability_gaussian'],
                                         GLOBAL_SETTINGS['probability_random_uniform']
                                         ])
        self._crossover_bins = np.cumsum([GLOBAL_SETTINGS['probability_SBX'],
                                          GLOBAL_SETTINGS['probability_SPBX']
                                          ])
        self._SPBX_type = GLOBAL_SETTINGS['SPBX_type'].lower()
        print(self.settings)
        self._mutation_rate = eval(self.settings['mutation_rate'])

        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = eval(self.settings['num_parents']) + eval(self.settings['num_offspring'])
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = eval(self.settings['num_offspring'])
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.settings['selection_type']))

        individuals: List[Individual] = []
        self.board_size = eval(self.settings["board_size"])

        for _ in range(eval(self.settings['num_parents'])):
            if create_snake:
                individual = create_snake()
            else:
                individual = Snake(self.board_size,
                                   hidden_layer_architecture=eval(self.settings['hidden_network_architecture']),
                                   hidden_activation=self.settings['hidden_layer_activation'],
                                   output_activation=self.settings['output_layer_activation'],
                                   lifespan=GLOBAL_SETTINGS['lifespan'],
                                   apple_and_self_vision=self.settings['apple_and_self_vision'])

            individuals.append(individual)
        self.t1 = time()

        self.best_fitness = 0
        self.best_score = 0

        self._current_individual = 0
        self.population = Population(individuals)

        self.snake = self.population.individuals[self._current_individual]
        self.myCanvas = NN_canvas(None, snake=self.snake, bg="white", height=1000, width=1000)
        self.current_generation = current_gen

        self.init_gen = self.current_generation
        self.times_to_save = times
        self.model_path = path

    def update(self, display=False) -> None:

        if self.snake.is_alive:
            self.snake.update()

        if self.snake.is_alive:

            self.snake.move()
            if display:
                self.myCanvas.update_network()

            if self.snake.score > self.best_score:
                self.best_score = self.snake.score


        else:

            self.snake.calculate_fitness()
            fitness = self.snake.fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness

            self._current_individual += 1

            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or \
                    (self.current_generation == self.init_gen and self._current_individual == eval(self.settings[
                                                                                                       'num_parents'])):
                print(self.settings)
                print('======================= Gneration {} at time {:.2f} ======================='.format(
                    self.current_generation, time() - self.t1))
                print('----Max fitness:', self.population.fittest_individual.fitness)
                print('----Best Score:', self.population.fittest_individual.score)
                print('----Average fitness:', self.population.average_fitness)
                print(f"----Wins: {WinCounter.counter / self._current_individual}")
                WinCounter.counter = 0
                if self.current_generation % self.times_to_save == 0 and self.current_generation > self.init_gen:
                    save_snake(self.model_path,
                               f"snake_{self.current_generation}_{self.population.fittest_individual.score:d}",
                               self.population.fittest_individual,
                               self.settings)
                save_stats(self.population, self.model_path, f"snake_stats")
                self.next_generation()
            else:
                current_pop = eval(
                    self.settings['num_parents']) if self.current_generation == 0 else self._next_gen_size

            self.snake = self.population.individuals[self._current_individual]
            self.myCanvas = NN_canvas(None, snake=self.snake, bg="white", height=1000, width=1000)

    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0

        for individual in self.population.individuals:
            individual.calculate_fitness()

        self.population.individuals = ellitism_selection(self.population, eval(self.settings['num_parents']))

        random.shuffle(self.population.individuals)
        next_pop: List[Snake] = []

        if self.settings['selection_type'].lower() == 'plus':

            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                params = individual.network.params
                board_size = individual.board_size
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                apple_and_self_vision = individual.apple_and_self_vision

                if lifespan > 0:
                    s = Snake(board_size, chromosome=params, hidden_layer_architecture=hidden_layer_architecture,
                              hidden_activation=hidden_activation, output_activation=output_activation,
                              lifespan=lifespan, apple_and_self_vision=apple_and_self_vision)
                    next_pop.append(s)

        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, 2)

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

            c1 = Snake(p1.board_size, chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                       hidden_activation=p1.hidden_activation, output_activation=p1.output_activation,
                       lifespan=GLOBAL_SETTINGS['lifespan'])
            c2 = Snake(p2.board_size, chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                       hidden_activation=p2.hidden_activation, output_activation=p2.output_activation,
                       lifespan=GLOBAL_SETTINGS['lifespan'])

            next_pop.extend([c1, c2])

        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _increment_generation(self):
        self.current_generation += 1

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias = SBX(parent1_bias, parent2_bias, self._SBX_eta)


        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights,
                                                                           major=self._SPBX_type)
            child1_bias, child2_bias = single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)

        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if self.settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)

        if mutation_bucket == 0:

            gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            gaussian_mutation(child2_bias, mutation_rate, scale=scale)


        elif mutation_bucket == 1:

            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')
