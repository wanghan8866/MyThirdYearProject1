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
import math
from Win_counter import WinCounter
angle_16 = math.tan(22.5 / 180 * math.pi)
class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')

    def __init__(self,
                 dist_to_wall: Union[float, int],
                 dist_to_apple: Union[float, int],
                 dist_to_self: Union[float, int]
                 ):
        self.dist_to_wall = float(dist_to_wall)
        self.dist_to_apple = float(dist_to_apple)
        self.dist_to_self = float(dist_to_self)


class DrawableVision(object):
    __slots__ = ('wall_location', 'apple_location', 'self_location')

    def __init__(self,
                 wall_location: Point,
                 apple_location: Optional[Point] = None,
                 self_location: Optional[Point] = None,
                 ):
        self.wall_location = wall_location
        self.apple_location = apple_location
        self.self_location = self_location

    def __str__(self):
        return f"{str(self.wall_location)} {str(self.apple_location)} {str(self.self_location)}"


class Snake(Individual,Env):
    def __init__(self, board_size: Tuple[int, int],
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 start_pos: Optional[Point] = None,
                 apple_seed: Optional[int] = None,
                 initial_velocity: Optional[str] = None,
                 starting_direction: Optional[str] = None,
                 hidden_layer_architecture: Optional[List[int]] = [1123125, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Optional[Union[int, float]] = np.inf,
                 apple_and_self_vision: Optional[str] = 'binary'
                 ):
        super(Snake, self).__init__()
        self.snake_array = None
        self.is_alive = True
        self.lifespan = lifespan
        self.apple_and_self_vision = apple_and_self_vision.lower()
        self.score = 0  
        self._fitness = 0  
        self._frames = 0  
        self._frames_since_last_apple = 0
        self.possible_directions = ('u', 'd', 'l', 'r')
        self.steps_history=[]

        self.board_size = board_size
        self.hidden_layer_architecture = hidden_layer_architecture

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if not start_pos:
            
            
            
            x = random.randint(2, self.board_size[0] - 3)
            y = random.randint(2, self.board_size[1] - 3)

            start_pos = Point(x, y)
        self.start_pos = start_pos

        self._vision_type = VISION_16
        self._vision: List[Vision] = [None] * len(self._vision_type)
        
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

        
        
        
        
        num_inputs = len(self._vision_type) * 3 + 4 + 4  
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]  
        self.network_architecture.extend(self.hidden_layer_architecture)  
        self.network_architecture.append(4)  
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                          )
        self.intersections=[]

        
        if chromosome:
            
            self.network.params = chromosome
            
        else:
            
            
            pass

        
        if apple_seed is None:
            apple_seed = np.random.randint(-1000000000, 1000000000)
        self.apple_seed = apple_seed  
        self.rand_apple = random.Random(self.apple_seed)

        self.apple_location = None
        if starting_direction:
            starting_direction = starting_direction[0].lower()
        else:
            starting_direction = self.possible_directions[random.randint(0, 3)]

        self.starting_direction = starting_direction  
        self.init_snake(self.starting_direction)
        self.initial_velocity = initial_velocity
        self.init_velocity(self.starting_direction, self.initial_velocity)
        self.generate_apple()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=self.vision_as_array.shape, dtype=np.float64)


    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self):
        
        self._fitness = (self._frames) + ((2 ** self.score) + (self.score ** 2.1) * 500) - (
                    ((.25 * self._frames) ** 1.3) * (self.score ** 1.2))
        
        self._fitness = max(self._fitness, .1)

    @property
    def chromosome(self):
        
        pass

    def encode_chromosome(self):
        
        
        
        
        
        
        
        pass

    def decode_chromosome(self):
        
        
        
        
        
        
        
        
        
        pass

    def look(self):
        
        self.intersections=[]
        for i, slope in enumerate(self._vision_type):
            x = self.snake_array[0].x
            y = self.snake_array[0].y
            dy1 = (y+1) * angle_16
            dy1 = 0 if dy1 < 1e-3 else dy1
            dy2 = abs(self.board_size[0] - y) * angle_16
            dy2 = 0 if dy2 < 1e-3 else dy2
            dx1 = (x+1) * angle_16
            dx1 = 0 if dx1 < 1e-3 else dx1
            dx2 = abs(self.board_size[0]- x) * angle_16
            dx2 = 0 if dx2 < 1e-3 else dx2

            if slope.rise == -2 and slope.run==1:
                vision, drawable_vision = self.look_in_direction_16((x + dy1, -1))
            elif slope.rise == -1 and slope.run==2:
                vision, drawable_vision = self.look_in_direction_16((self.board_size[0], y - dx2))
            elif slope.rise == 1 and slope.run == 2:
                vision, drawable_vision = self.look_in_direction_16((self.board_size[0], y + dx2))
            elif slope.rise == 2 and slope.run == 1:
                vision, drawable_vision = self.look_in_direction_16((x+dy2, self.board_size[0] ))
            elif slope.rise == 2 and slope.run == -1:
                vision, drawable_vision = self.look_in_direction_16((x-dy2, self.board_size[0] ))
            elif slope.rise == 1 and slope.run == -2:
                vision, drawable_vision = self.look_in_direction_16((-1, y + dx1))
            elif slope.rise == -1 and slope.run == -2:
                vision, drawable_vision = self.look_in_direction_16((-1, y - dx1))
            elif slope.rise == -2 and slope.run == -1:
                vision, drawable_vision = self.look_in_direction_16((x - dy1, -1))
            else:
                vision, drawable_vision = self.look_in_direction(slope)
            self._vision[i] = vision
            self._drawable_vision[i] = drawable_vision
            




        
        self._vision_as_input_array()

    def look_in_direction_16(self, new_pos:Tuple[int, int])->Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.snake_array[0].copy()
        distance = 1.0
        total_distance = 1.0
        x = position.x
        y = position.y
        x2 = new_pos[0]
        y2 = new_pos[1]

        points = findPoints(x, y, x2, y2)


        
        body_found = False  
        food_found = False  
        total_distance += distance
        
        for point in points[1:]:
            position.x = point[0]
            position.y = point[1]
            wall_location = position
            if not self._within_wall(position):
                break
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

    def _vision_as_input_array(self) -> None:
        
        
        for va_index, v_index in zip(range(0, len(self._vision) * 3, 3), range(len(self._vision))):
            if self._vision[v_index] == None:
                continue
            vision = self._vision[v_index]
            self.vision_as_array[va_index, 0] = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.dist_to_apple
            self.vision_as_array[va_index + 2, 0] = vision.dist_to_self

        i = len(self._vision) * 3  

        direction = self.direction[0].lower()
        
        direction_one_hot = np.zeros((len(self.possible_directions), 1))
        direction_one_hot[self.possible_directions.index(direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = direction_one_hot

        i += len(self.possible_directions)

        
        tail_direction_one_hot = np.zeros((len(self.possible_directions), 1))
        tail_direction_one_hot[self.possible_directions.index(self.tail_direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = tail_direction_one_hot

    def _within_wall(self, position: Point) -> bool:
        return position.x >= 0 and position.y >= 0 and \
               position.x < self.board_size[0] and \
               position.y < self.board_size[1]

    def generate_apple(self) -> None:
        width = self.board_size[0]
        height = self.board_size[1]
        
        possibilities = [divmod(i, height) for i in range(width * height) if
                         divmod(i, height) not in self._body_locations]
        if possibilities:
            loc = self.rand_apple.choice(possibilities)
            self.apple_location = Point(loc[0], loc[1])
        else:
            
            print('you won!')
            WinCounter.counter+=1


    def init_snake(self, starting_direction: str) -> None:
        """
        Initialize teh snake.
        starting_direction: ('u', 'd', 'l', 'r')
            direction that the snake should start facing. Whatever the direction is, the head
            of the snake will begin pointing that way.
        """
        head = self.start_pos
        
        if starting_direction == 'u':
            snake = [head, Point(head.x, head.y + 1), Point(head.x, head.y + 2)]
        
        elif starting_direction == 'd':
            snake = [head, Point(head.x, head.y - 1), Point(head.x, head.y - 2)]
        
        elif starting_direction == 'l':
            snake = [head, Point(head.x + 1, head.y), Point(head.x + 2, head.y)]
        
        elif starting_direction == 'r':
            snake = [head, Point(head.x - 1, head.y), Point(head.x - 2, head.y)]

        self.snake_array = deque(snake)
        self._body_locations = set(snake)
        self.is_alive = True

    def update(self, _action=-1):
        if self.is_alive:
            self._frames += 1
            self.look()
            if _action not in self.possible_directions :
                
                self.network.feed_forward(self.vision_as_array)
                self.direction = self.possible_directions[np.argmax(self.network.out)]
            else:
                print("update",_action)
                self.direction = _action

            return True
        else:
            return False

    def move(self) -> bool:
        if not self.is_alive:
            return False

        direction = self.direction[0].lower()
        
        if direction not in self.possible_directions:
            return False

        
        
        head = self.snake_array[0]
        if direction == 'u':
            next_pos = Point(head.x, head.y - 1)
        elif direction == 'd':
            next_pos = Point(head.x, head.y + 1)
        elif direction == 'r':
            next_pos = Point(head.x + 1, head.y)
        elif direction == 'l':
            next_pos = Point(head.x - 1, head.y)

        
        if self._is_valid(next_pos):
            
            if next_pos == self.snake_array[-1]:
                
                
                self.snake_array.pop()
                self.snake_array.appendleft(next_pos)
                
            elif next_pos == self.apple_location:
                self.score += 1

                
                self.snake_array.appendleft(next_pos)
                self._body_locations.update({next_pos})
                
                self.generate_apple()
                distance = abs(self.apple_location.x - next_pos.x) + abs(self.apple_location.y - next_pos.y)
                self.steps_history.append(distance/self._frames_since_last_apple)
                self._frames_since_last_apple = 0
            
            else:
                
                self.snake_array.appendleft(next_pos)
                self._body_locations.update({next_pos})
                
                tail = self.snake_array.pop()
                self._body_locations.symmetric_difference_update({tail})

            
            p2 = self.snake_array[-2]
            p1 = self.snake_array[-1]
            diff = p2 - p1
            if diff.x < 0:
                self.tail_direction = 'l'
            elif diff.x > 0:
                self.tail_direction = 'r'
            elif diff.y > 0:
                self.tail_direction = 'd'
            elif diff.y < 0:
                self.tail_direction = 'u'

            self._frames_since_last_apple += 1
            
            if self._frames_since_last_apple > self.board_size[0]*self.board_size[1]*2:
                self.is_alive = False
                return False

            return True
        else:
            self.is_alive = False
            return False

    def _is_apple_location(self, position: Point) -> bool:
        return position == self.apple_location

    def _is_body_location(self, position: Point) -> bool:
        return position in self._body_locations

    def _is_valid(self, position: Point) -> bool:
        """
        Determine whether a given position is valid.
        Return True if the position is on the board and does not intersect the snake.
        Return False otherwise
        """
        if (position.x < 0) or (position.x > self.board_size[0] - 1):
            return False
        if (position.y < 0) or (position.y > self.board_size[1] - 1):
            return False

        if position == self.snake_array[-1]:
            return True
        
        
        elif position in self._body_locations:
            return False
        
        else:
            return True

    def init_velocity(self, starting_direction, initial_velocity: Optional[str] = None) -> None:
        if initial_velocity:
            self.direction = initial_velocity[0].lower()
        
        else:
            self.direction = starting_direction

        
        self.tail_direction = self.direction

    def step(self, action):
        self.update(action)
        self.move()
        info ={}

        return self.vision_as_array, self.score, not self.is_alive, info

    def reset(self):
        self.score = 0  
        self._fitness = 0  
        self._frames = 0  
        self._frames_since_last_apple = 0
        self.is_alive=True

        
        
        x = random.randint(2, self.board_size[0] - 3)
        y = random.randint(2, self.board_size[1] - 3)
        self.steps_history=[]
        self.start_pos = Point(x, y)
        self.apple_location = None
        self.starting_direction = self.possible_directions[random.randint(0, 3)]  
        self.init_snake(self.starting_direction)
        self.init_velocity(self.starting_direction, self.initial_velocity)
        self.generate_apple()
        self.intersections=[]
        print(self.apple_seed)

        self.update()

        return self.vision_as_array
    def render(self, mode="human"):
        size = 40
        self.look()
        print()
        for i in range(16):
            print([a[0] for a in self.vision_as_array[3*i:3*(i+1)]])

        self.img = np.zeros((self.board_size[0] * size, self.board_size[0] * size, 3), dtype='uint8')
        cv2.rectangle(self.img, (self.apple_location.x * size, self.apple_location.y * size),
                      (self.apple_location.x * size + size, self.apple_location.y  * size + size), (0, 0, 255), -1, lineType = cv2.LINE_AA)

        
        
        half = size//2
        for vision in self._drawable_vision:
            if vision == None:
                continue
            
            colour =  (255, 100, 100)
            if vision.apple_location:
                colour = (50, 255, 60)
            elif vision.self_location:
                colour = (10,10,255)
            
            if vision.wall_location is None:
                continue
            
            dx=0
            dy=0
            cv2.line(self.img,
                     (self.snake_array[0].x * size + half, self.snake_array[0].y * size +half),
                     ((vision.wall_location.x-dx) * size + half, (vision.wall_location.y-dy) * size+half ),
                    colour, thickness=1, lineType = cv2.LINE_AA)
        for point in self.intersections:
            if point is None:
                continue
            cv2.circle(self.img, ((point.x -dx) *size+ half, (point.y-dy)*size +half), 5, (100,255,50), 1, lineType=cv2.LINE_AA)
        for position in self.snake_array:
            cv2.rectangle(self.img, (position.x * size, position.y * size),
                          (position.x * size + size, position.y * size + size), (0, 255, 0), 3, lineType = cv2.LINE_AA)
        cv2.rectangle(self.img, (self.snake_array[0].x * size, self.snake_array[0].y * size),
                      (self.snake_array[0].x * size + size, self.snake_array[0].y * size + size), (200, 0, 125), 3, lineType = cv2.LINE_AA)
        if mode == "human":
            cv2.imshow('a', self.img)
        else:
            return self.img



def save_snake(population_folder: str, individual_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
    print("...Save...")
    
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    
    
    
    
    constructor = {}
    constructor['start_pos'] = snake.start_pos.to_dict()
    constructor['apple_seed'] = snake.apple_seed
    constructor['initial_velocity'] = snake.initial_velocity
    constructor['starting_direction'] = snake.starting_direction
    snake_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

    
    with open(snake_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(snake.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = snake.network.params[w_name]
        bias = snake.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)


def load_snake(population_folder: str, individual_name: str,
               settings: Optional[Union[Dict[str, Any], str]] = None) -> Snake:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception(
                "settings needs to be passed as an argument if 'settings.json' does not exist under population folder")

        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        filepath = settings
        with open(filepath, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    
    constructor_params = {}
    snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(snake_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    snake = Snake(settings['board_size'], chromosome=params,
                  start_pos=Point.from_dict(constructor_params['start_pos']),
                  apple_seed=constructor_params['apple_seed'],
                  initial_velocity=constructor_params['initial_velocity'],
                  starting_direction=constructor_params['starting_direction'],
                  hidden_layer_architecture=settings['hidden_network_architecture'],
                  hidden_activation=settings['hidden_layer_activation'],
                  output_activation=settings['output_layer_activation'],
                  lifespan=settings['lifespan'],
                  apple_and_self_vision=settings['apple_and_self_vision']
                  )
    return snake
from settings import settings
PRINT_NUM = 10
if __name__ == '__main__':
    
    
    
    
    
    
    env = load_snake("models/test_64", f"snake_1500", settings)
    
    
    episodes = 20
    rewards_history = []
    avg_reward_history = []
    print(env.observation_space.shape)
    for episode in range(episodes):
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            action = env.possible_directions[env.action_space.sample()]
            t_end = time.time() + 0.5
            k = -1
            
            while time.time() < t_end:
                if k == -1:
                    
                    k = cv2.waitKey(1)
                    

                    if k == 97:
                        action = "l"
                    elif k == 100:
                        action = "r"
                    elif k == 119:
                        action = "u"
                    elif k == 115:
                        action = "d"
                    
                else:
                    continue
            
            action=-1

            obs, reward, done, info = env.step(action)
            env.render()

            
            
            
            
            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        print(env.steps_history)
        print(f"snake: {len(env.snake_array)}")
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    plt.plot(rewards_history)
    plt.plot(avg_reward_history)
    plt.show()
