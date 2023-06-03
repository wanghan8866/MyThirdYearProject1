import collections
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import List
from typing import Tuple

import cv2
import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from matplotlib import pyplot as plt

from memory import MaxHeap
from network import LinearDeepQNetwork, DeepQNetwork
from snake_env2 import SnakeEnv2


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


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

        self.apple_num = 0

    def step(self, action):
        # self.prev_actions.append(action)
        # cv2.imshow('a', self.img)
        # cv2.waitKey(1)
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

        # Takes step after fixed time
        # t_end = time.time() + 0.25
        # k = -1
        # # action = -1
        # while time.time() < t_end:
        #     if k == -1:
        #         # pass
        #         k = cv2.waitKey(1)
        #         # print(k)
        #
        #         if k == 97:
        #             action = Direction.LEFT
        #         elif k == 100:
        #             action = Direction.RIGHT
        #         elif k == 119:
        #             action = Direction.UP
        #         elif k == 115:
        #             action = Direction.DOWN
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
            apple_reward = 2
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
        self.total_reward = apple_reward + death_reward

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
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.prev_reward = 0

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

    def render(self, mode='human'):
        if mode == "human":
            # cv2.waitKey(1)
            cv2.imshow('a', self.img)
            # print(self.observation)
        else:
            return self.img


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(LinearDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        print(input_dims)
        # self.fc1 = nn.Linear(*input_dims, 32)
        # self.fc2 = nn.Linear(32, 32)
        # self.q = nn.Linear(32, n_actions)
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.q = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc2(flat2))
        q = self.q(flat3)

        return q

    def save_checkpoint(self):
        print("... save ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... load ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv1d(input_dims[0], 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 3)
        # self.conv3 = nn.Conv1d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        #
        # self.fc1 = nn.Linear(fc_input_dims, 512)
        # self.fc2 = nn.Linear(512, n_actions)
        print(fc_input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        # conv3 = F.relu(self.conv3(conv2))
        conv_state = conv2.view(conv2.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        q = self.fc2(flat1)

        return q

    def calculate_conv_output_dims(self, input_dims):
        # print(input_dims)
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        # dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


@dataclass
class MemoryCell:
    priority: float
    rank: int
    transition: List[np.array] = field(repr=False)

    def update_priority(self, new_priority: float):
        self.priority = new_priority

    def update_rank(self, new_rank: int):
        self.rank = new_rank

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority < other.priority


class MaxHeap:
    def __init__(self, max_size: int = 1e6, n_batches: int = 32,
                 alpha: float = 0.5, beta: float = 0, r_iter: int = 32):
        self.array: List[MemoryCell] = []
        self.max_size = max_size
        self.mem_cntr: int = 0
        self.n_batches = n_batches
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.alpha_start = alpha
        self.r_iter = r_iter
        self._precompute_indices()

    def store_transition(self, sarsd: List[np.array]):
        priority = 10
        rank = 1
        transition = MemoryCell(priority, rank, sarsd)
        self._insert(transition)

    def _insert(self, transition: MemoryCell):
        if self.mem_cntr < self.max_size:
            self.array.append(transition)
        else:
            index = self.mem_cntr % self.max_size
            self.array[index] = transition
        self.mem_cntr += 1

    def _update_ranks(self):
        array = deepcopy(self.array)
        indices = [i for i in range(len(array))]
        sorted_array = [list(x) for x in zip(*sorted(zip(array, indices),
                                                     key=lambda pair: pair[0],
                                                     reverse=True))]

        for index, value in enumerate(sorted_array[1]):
            self.array[value].rank = index + 1

    def print_array(self, a=None):
        array = self.array if a is None else a
        for cell in array:
            print(cell)
        print('\n')

    def _max_heapify(self, array: List[MemoryCell], i: int, N: int = None):
        N = len(array) if N is None else N
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < N and array[left] > array[i]:
            largest = left
        if right < N and array[right] > array[largest]:
            largest = right
        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            self._max_heapify(array, largest, N)
        return array

    def _build_max_heap(self):
        array = deepcopy(self.array)
        N = len(array)
        for i in range(N // 2, -1, -1):
            array = self._max_heapify(array, i)
        return array

    def rebalance_heap(self):
        self.array = self._build_max_heap()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, index in enumerate(indices):
            self.array[index].update_priority(priorities[idx])

    def ready(self):
        return self.mem_cntr >= self.n_batches

    def anneal_beta(self, ep: int, ep_max: int):
        self.beta = self.beta_start + ep / ep_max * (1 - self.beta_start)

    def anneal_alpha(self, ep: int, ep_max: int):
        self.alpha = self.alpha_start * (1 - ep / ep_max)

    def _precompute_indices(self):
        print('precomputing indices')
        self.indices = []
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        start = [i for i in range(n_batches, self.max_size + 1, n_batches)]
        for start_idx in start:
            bs = start_idx // n_batches
            indices = np.array([[j * bs + k for k in range(bs)]
                                for j in range(n_batches)], dtype=np.int16)
            self.indices.append(indices)

    def compute_probs(self):
        self.probs = []
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        idx = min(self.mem_cntr, self.max_size) // n_batches - 1
        for indices in self.indices[idx]:
            probs = []
            for index in indices:
                p = 1 / (self.array[index].rank) ** self.alpha
                probs.append(p)
            z = [p / sum(probs) for p in probs]
            self.probs.append(z)

    def _calculate_weights(self, probs: List):
        weights = np.array([(1 / self.mem_cntr * 1 / prob) ** self.beta
                            for prob in probs])
        weights *= 1 / (max(weights))
        return weights

    def sample(self):
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        idx = min(self.mem_cntr, self.max_size) // n_batches - 1
        if self.r_iter != 1:
            samples = [np.random.choice(self.indices[idx][row],
                                        p=self.probs[row])
                       for row in range(len(self.indices[idx]))]
            p = [val for row in self.probs for val in row]
            probs = [p[s] for s in samples]
        else:
            samples = np.random.choice(self.indices[idx][0], self.n_batches)
            probs = [1 / len(samples) for _ in range(len(samples))]
        weights = self._calculate_weights(probs)
        mems = np.array([self.array[s] for s in samples])
        sarsd = []
        for item in mems:
            row = []
            for i in range(len(item.transition)):
                row.append(np.array(item.transition[i]))
            sarsd.append(row)
        return sarsd, samples, weights


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, alpha=0.5, beta=0, r_iter=32,
                 algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.rebalance_iter = r_iter

        self.memory = MaxHeap(mem_size, batch_size, alpha=alpha, beta=beta,
                              r_iter=r_iter)

        # self.q_eval = DeepQNetwork(self.lr, self.n_actions,
        #                             input_dims=self.input_dims,
        #                             name=self.env_name+'_'+self.algo+'_q_eval',
        #                             chkpt_dir=self.chkpt_dir)
        #
        # self.q_next = DeepQNetwork(self.lr, self.n_actions,
        #                             input_dims=self.input_dims,
        #                             name=self.env_name+'_'+self.algo+'_q_next',
        #                             chkpt_dir=self.chkpt_dir)
        self.q_eval = LinearDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=self.env_name + '_' + self.algo + '_q_eval',
                                         chkpt_dir=self.chkpt_dir)

        self.q_next = LinearDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=self.env_name + '_' + self.algo + '_q_next',
                                         chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition([state, action, reward, state_, done])

    def sample_memory(self):
        sarsd, sample_idx, weights = self.memory.sample()
        states = np.array([row[0] for row in sarsd])
        actions = np.array([row[1] for row in sarsd])
        rewards = np.array([row[2] for row in sarsd])
        states_ = np.array([row[3] for row in sarsd])
        dones = np.array([row[4] for row in sarsd])

        states = T.tensor(states, dtype=T.float).to(self.q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions, dtype=T.long).to(self.q_eval.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.q_eval.device)

        weights = T.tensor(weights, dtype=T.float).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, sample_idx, weights

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def rebalance_heap(self):
        if self.rebalance_iter > 1:
            if self.learn_step_counter % self.rebalance_iter == 0:
                self.memory.rebalance_heap()
                self.memory._update_ranks()
                self.memory.compute_probs()

    def learn(self):
        if not self.memory.ready():
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        self.rebalance_heap()

        states, actions, rewards, states_, dones, \
        sample_idx, weights = self.sample_memory()
        indices = np.arange(self.batch_size)
        # print(indices)
        # print(actions)
        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        td_error = np.abs((q_target.detach().cpu().numpy() -
                           q_pred.detach().cpu().numpy()))
        td_error = np.clip(td_error, -1., 1.)

        self.memory.update_priorities(sample_idx, td_error)

        q_target *= weights
        q_pred *= weights

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


def clip_reward(r):
    if r > 1:
        return 1
    elif r < -1:
        return -1
    else:
        return r


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        # print(np.array([env.observation_space.low]).repeat(repeat, axis=0).shape)
        # print(env.observation_space.high.repeat(repeat, axis=0))
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),

            dtype=np.float32)
        # print(self.observation_space.low.shape)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()

        observation = self.env.reset()

        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        # print(np.array(self.stack, dtype=np.float16).shape)
        return np.array(self.stack, dtype=np.float16).reshape(
            self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env_id = "snake-v4"
    env = SnakeEnv2(10)
    print(env.observation_space.shape)
    env = StackFrames(env, repeat=50)
    print(env.observation_space.shape)
    best_score = -np.inf
    load_checkpoint = False
    n_games = int(1e6)
    max_frames = int(1e6)
    game_to_save = 0.5 * max_frames
    r_iter = 64
    alpha = 0.25
    beta = 0.5
    replace = 250
    N_Print = 300

    if load_checkpoint:
        epsilon = 0
    else:
        epsilon = 1
    agent = DQNAgent(gamma=0.99, epsilon=epsilon, lr=2.5e-4, alpha=alpha,
                     beta=beta, r_iter=r_iter,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50 * 1024,
                     eps_min=0.01,
                     batch_size=64, replace=replace, eps_dec=1e-4,
                     chkpt_dir='models/', algo='DQNAgent_ranked',
                     env_name=env_id)

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games' + str(alpha) + \
            'alpha_' + str(beta) + '_replace_' + str(replace)
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing,
    # do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    # env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array, apples, frames = [], [], [], [], []

    t1 = time()
    best_apple = 0
    best_frames = 0
    average_frames = 0
    average_score = 0
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        info = {}
        frames_per_game = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            #
            # r = clip_reward(reward)
            # print(np.sum(observation), r)
            # print(observation_.shape)
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()
            else:
                env.render(mode="human")
            observation = observation_
            n_steps += 1
            frames_per_game += 1
            # print(n_steps)
        scores.append(score)
        steps_array.append(n_steps)
        apples.append(info["apple score"])
        # print(info["apple score"])
        frames.append(frames_per_game)

        avg_score = np.mean(scores[-100:])

        if i % N_Print == 0:
            t = (time() - t1) / n_steps * (max_frames - n_steps) / 60. / 60.
            print('episode: ', i, "steps:", n_steps, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                  'epsilon %.2f' % agent.epsilon,
                  f"time: {time() - t1}, remaining time: {t} h, apple: {best_apple}, frames:{best_frames}")

        if avg_score > best_score:
            if not load_checkpoint and agent.epsilon < 0.05:
                agent.save_models()
            best_score = avg_score
            best_apple = np.mean(apples[-100:])
            best_frames = np.mean(frames[-100:])

        eps_history.append(agent.epsilon)
        if n_steps > max_frames:
            break

    x = [i + 1 for i in range(len(scores))]
    np.save(f"models/{env_id}_DQNAgent_x", x)
    np.save(f"models/{env_id}_DQNAgent_scores", scores)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
