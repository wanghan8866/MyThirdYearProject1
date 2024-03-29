from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake_env import *
import numpy as np
from nn_viz import NeuralNetworkViz
from neural_network import FeedForwardNetwork, sigmoid, linear, relu
from settings import settings
from genetic_algorithm.population import Population
from genetic_algorithm.selection import ellitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover
from math import sqrt
from decimal import Decimal
import random
import csv
from snakes import Snakes
from time import time
from Win_counter import WinCounter

SQUARE_SIZE = (35, 35)

t1 = time()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, show=False, fps=200):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.setPalette(palette)
        self.settings = settings
        self._SBX_eta = self.settings['SBX_eta']
        self._mutation_bins = np.cumsum([self.settings['probability_gaussian'],
                                         self.settings['probability_random_uniform']
                                         ])
        self._crossover_bins = np.cumsum([self.settings['probability_SBX'],
                                          self.settings['probability_SPBX']
                                          ])
        self._SPBX_type = self.settings['SPBX_type'].lower()
        self._mutation_rate = self.settings['mutation_rate']

        
        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = self.settings['num_parents'] + self.settings['num_offspring']
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = self.settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.settings['selection_type']))

        self.board_size = settings['board_size']
        self.border = (0, 10, 0, 10)  
        self.snake_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.snake_widget_height = SQUARE_SIZE[1] * self.board_size[1]

        
        self._snake_widget_width = max(self.snake_widget_width, 620)
        self._snake_widget_height = max(self.snake_widget_height, 600)

        self.top = 150
        self.left = 150
        self.width = self._snake_widget_width + 700 + self.border[0] + self.border[2]
        self.height = self._snake_widget_height + self.border[1] + self.border[3] + 200

        individuals: List[Individual] = []

        for _ in range(self.settings['num_parents']):
            
            
            
            
            
            
            individual = Snakes(settings["board_size"],
                                "models/test_64",
                                f"snake_1699",
                                number_rounds=25,
                                loading=False
                                )
            individuals.append(individual)

        self.best_fitness = 0
        self.best_score = 0

        self._current_individual = 0
        self.population = Population(individuals)

        self.snake = self.population.individuals[self._current_individual]
        self.current_generation = 0
        
        self.init_gen = self.current_generation

        

        
        
        

        if show:
            self.show()


    def update(self) -> None:
        
        
        if self.snake.is_alive:
            self.snake.update()
        
        
        if self.snake.is_alive:
            
            self.snake.move()
            
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score
                
        
        else:
            
            self.snake.calculate_fitness()
            fitness = self.snake.fitness
            

            
            
            
            
            

            
            
            
            
            

            
            
            
            

            

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                

            self._current_individual += 1
            
            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or \
                    (self.current_generation == self.init_gen and self._current_individual == settings['num_parents']):
                print(self.settings)
                print('======================= Gneration {} at time {:.2f} ======================='.format(
                    self.current_generation, time() - t1))
                print('----Max fitness:', self.population.fittest_individual.fitness)
                print('----Best Score:', self.population.fittest_individual.score)
                print('----Average fitness:', self.population.average_fitness)
                print(f"----Wins: {WinCounter.counter}, {WinCounter.counter / (self.population.fittest_individual.number_rounds * self._next_gen_size)}")
                WinCounter.counter = 0
                if self.current_generation % 1 == 0 and self.current_generation > self.init_gen:
                    save_snake("models/snakes_64", f"snake_{self.current_generation}_{self.population.fittest_individual.score:d}", self.population.fittest_individual,
                               settings)
                save_stats(self.population, "models/snakes_64", f"snake_stats")
                self.next_generation()
            else:
                current_pop = self.settings['num_parents'] if self.current_generation == 0 else self._next_gen_size
                
                

            
            self.snake = self.population.individuals[self._current_individual]
            
            

    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0

        
        for individual in self.population.individuals:
            individual.calculate_fitness()

        self.population.individuals = ellitism_selection(self.population, self.settings['num_parents'])

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
                    s = Snakes(board_size, chromosome=params, hidden_layer_architecture=hidden_layer_architecture,
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

            
            c1 = Snakes(p1.board_size, chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                        hidden_activation=p1.hidden_activation, output_activation=p1.output_activation,
                        lifespan=self.settings['lifespan'])
            c2 = Snakes(p2.board_size, chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                        hidden_activation=p2.hidden_activation, output_activation=p2.output_activation,
                        lifespan=self.settings['lifespan'])

            
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
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        
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


class GeneticAlgoWidget(QtWidgets.QWidget):
    def __init__(self, parent, settings):
        super().__init__(parent)

        font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
        font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 5)
        TOP_LEFT = Qt.AlignLeft | Qt.AlignTop

        LABEL_COL = 0
        STATS_COL = 1
        ROW = 0

        
        
        self._create_label_widget_in_grid('Generation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_generation_label = self._create_label_widget('1', font)
        grid.addWidget(self.current_generation_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Individual: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_individual_label = self._create_label_widget('1/{}'.format(settings['num_parents']), font)
        grid.addWidget(self.current_individual_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Best Score: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.best_score_label = self._create_label_widget('0', font)
        grid.addWidget(self.best_score_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Best Fitness: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.best_fitness_label = self._create_label_widget('{:.2E}'.format(Decimal('0.1')), font)
        grid.addWidget(self.best_fitness_label, ROW, STATS_COL, TOP_LEFT)

        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        
        self._create_label_widget_in_grid('GA Settings', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        
        selection_type = ' '.join([word.lower().capitalize() for word in settings['selection_type'].split('_')])
        self._create_label_widget_in_grid('Selection Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(selection_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        prob_SBX = settings['probability_SBX']
        prob_SPBX = settings['probability_SPBX']
        crossover_type = '{:.0f}% SBX\n{:.0f}% SPBX'.format(prob_SBX * 100, prob_SPBX * 100)
        self._create_label_widget_in_grid('Crossover Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(crossover_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        prob_gaussian = settings['probability_gaussian']
        prob_uniform = settings['probability_random_uniform']
        mutation_type = '{:.0f}% Gaussian\t\n{:.0f}% Uniform'.format(prob_gaussian * 100, prob_uniform * 100)
        self._create_label_widget_in_grid('Mutation Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(mutation_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Mutation Rate:', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        mutation_rate_percent = '{:.0f}%'.format(settings['mutation_rate'] * 100)
        mutation_rate_type = settings['mutation_rate_type'].lower().capitalize()
        mutation_rate = mutation_rate_percent + ' + ' + mutation_rate_type
        self._create_label_widget_in_grid(mutation_rate, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Lifespan: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        lifespan = str(settings['lifespan']) if settings['lifespan'] != np.inf else 'infinite'
        self._create_label_widget_in_grid(lifespan, font, grid, ROW, STATS_COL, TOP_LEFT)

        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        
        self._create_label_widget_in_grid('NN Settings', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        
        hidden_layer_activation = ' '.join(
            [word.lower().capitalize() for word in settings['hidden_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Hidden Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(hidden_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        output_layer_activation = ' '.join(
            [word.lower().capitalize() for word in settings['output_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Output Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(output_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        network_architecture = '[{}, {}, 4]'.format(settings['vision_type'] * 3 + 4 + 4,
                                                    ', '.join([str(num_neurons) for num_neurons in
                                                               settings['hidden_network_architecture']]))
        self._create_label_widget_in_grid('NN Architecture: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(network_architecture, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        snake_vision = str(settings['vision_type']) + ' directions'
        self._create_label_widget_in_grid('Snake Vision: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(snake_vision, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        
        self._create_label_widget_in_grid('Apple/Self Vision: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        apple_self_vision_type = settings['apple_and_self_vision'].lower()
        self._create_label_widget_in_grid(apple_self_vision_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 2)

        self.setLayout(grid)

        self.show()

    def _create_label_widget(self, string_label: str, font: QtGui.QFont) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0, 0, 0, 0)
        return label

    def _create_label_widget_in_grid(self, string_label: str, font: QtGui.QFont,
                                     grid: QtWidgets.QGridLayout, row: int, col: int,
                                     alignment: Qt.Alignment) -> None:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(label, row, col, alignment)


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), snake=None):
        super().__init__(parent)
        self.board_size = board_size
        
        
        if snake:
            self.snake = snake
        self.setFocus()

        self.draw_vision = False
        self.show()

    def new_game(self) -> None:
        self.snake = Snakes(self.board_size)

    def update(self):
        if self.snake.is_alive:
            self.snake.update()
            self.repaint()
        else:
            
            pass

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.black)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_snake(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(Qt.red)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(198, 5, 20)))

        

        def _draw_line_to_apple(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> \
                Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.green))
            end_x = drawable_vision.apple_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0] / 2
            end_y = drawable_vision.apple_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1] / 2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        def _draw_line_to_self(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> \
                Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.red))
            end_x = drawable_vision.self_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0] / 2
            end_y = drawable_vision.self_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1] / 2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        for point in self.snake.snake_array:
            painter.drawRect(point.x * SQUARE_SIZE[0],  
                             point.y * SQUARE_SIZE[1],  
                             SQUARE_SIZE[0],  
                             SQUARE_SIZE[1])  

        if self.draw_vision:
            start = self.snake.snake_array[0]

            if self.snake._drawable_vision[0]:
                for drawable_vision in self.snake._drawable_vision:
                    start_x = start.x * SQUARE_SIZE[0] + SQUARE_SIZE[0] / 2
                    start_y = start.y * SQUARE_SIZE[1] + SQUARE_SIZE[1] / 2
                    if drawable_vision.apple_location and drawable_vision.self_location:
                        apple_dist = self._calc_distance(start.x, drawable_vision.apple_location.x, start.y,
                                                         drawable_vision.apple_location.y)
                        self_dist = self._calc_distance(start.x, drawable_vision.self_location.x, start.y,
                                                        drawable_vision.self_location.y)
                        if apple_dist <= self_dist:
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                        else:
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.apple_location:
                        start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.self_location:
                        start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)

                    if drawable_vision.wall_location:
                        painter.setPen(QtGui.QPen(Qt.black))
                        end_x = drawable_vision.wall_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0] / 2
                        end_y = drawable_vision.wall_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1] / 2
                        painter.drawLine(start_x, start_y, end_x, end_y)

    def draw_apple(self, painter: QtGui.QPainter) -> None:
        apple_location = self.snake.apple_location
        if apple_location:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))

            painter.drawRect(apple_location.x * SQUARE_SIZE[0],
                             apple_location.y * SQUARE_SIZE[1],
                             SQUARE_SIZE[0],
                             SQUARE_SIZE[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_apple(painter)
        self.draw_snake(painter)

        painter.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key_press = event.key()
        
        
        
        
        
        
        
        

    def _calc_distance(self, x1, x2, y1, y2) -> float:
        diff_x = float(abs(x2 - x1))
        diff_y = float(abs(y2 - y1))
        dist = ((diff_x * diff_x) + (diff_y * diff_y)) ** 0.5
        return dist


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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    while True:
        window.update()
    sys.exit(app.exec_())
