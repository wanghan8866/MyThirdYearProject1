from training.base_training_env import BaseTrainingEnv
import gym
import numpy as np
from PER.ranked.agent import DQNAgent
# from PER.ranked.snake_env5 import SnakeEnv5
from time import time, sleep
import cv2
import collections
from PER.ranked.deepQ_nn_vis import Q_NN_canvas
from PER.ranked.snake_env2 import SnakeEnv2
import csv
from typing import List, Union, Tuple
import os


def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)


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
        # self.observation_space = gym.spaces.Box(
        #     np.array([env.observation_space.low]).repeat(repeat, axis=0),
        #     np.array([env.observation_space.high]).repeat(repeat, axis=0),
        #
        #     dtype=np.uint8)
        self._frames = env._frames
        self.observation_space = gym.spaces.Box(
            np.array(env.observation_space.low).repeat(repeat, axis=0),
            np.array(env.observation_space.high).repeat(repeat, axis=0),

            dtype=np.uint8)
        print(self.observation_space.low.shape)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()

        observation = self.env.reset()

        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        # print(np.array(self.stack, dtype=np.float16).shape)
        return np.array(self.stack, dtype=np.uint8).reshape(
            self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


class Pop:
    def __init__(self, agent):
        self.fittest_individual = agent
        self.average_fitness = 0

    def setIndividual(self, agent):
        self.fittest_individual = agent


def save_stats(snake, path_to_dir: str, fname: str):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    f = os.path.join(path_to_dir, fname + '.csv')

    frames = [individual._frames for individual in [snake]]
    apples = [individual.score for individual in [snake]]
    fitness = [individual.fitness for individual in [snake]]
    # steps_history = [individual.steps_history for individual in population.individuals][np.argmax(apples)]

    write_header = True
    if os.path.exists(f):
        write_header = False

    trackers = [('steps', frames),
                ('apples', apples),
                ('fitness', fitness)
                ]
    stats = ['mean', 'median', 'std', 'min', 'max']
    # +[f"steps_{i}" for i in range(len(steps_history))]
    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(f, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}
        # Create a row to insert into csv
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat
        # for i in range(len(steps_history)):
        #     entry_name=f"steps_{i}"
        #     row[entry_name] = steps_history[i]

        # Write row
        writer.writerow(row)


class DQNTrainingEnv(BaseTrainingEnv):
    def __init__(self, setting, *args, **kwargs):
        super().__init__(setting, *args, **kwargs)
        env_id = "snake-v2-r4"
        # env = Snake([10,10])
        self.env = SnakeEnv2(10)

        print(self.env.observation_space.shape)
        self.env = StackFrames(self.env, repeat=4)
        self.snake = self.env
        print(self.env.observation_space.shape)
        self.best_score = -np.inf
        self.load_checkpoint = False
        self.testing = False
        self.n_games = int(1e6)
        self.max_frames = int(1e6)
        self.game_to_save = 0.5 * self.max_frames
        r_iter = 64
        alpha = 0.25
        beta = 0.5
        replace = 250
        self.N_Print = 300
        self._current_individual = 1
        if self.testing:
            n_games = 100
            self.test_scores = []

        if self.load_checkpoint:
            epsilon = 0
        else:
            epsilon = 1
        self.model_path = "C:/Users/w9264/MyThirdYearProject1/models/DQ/test1"
        self.agent = DQNAgent(gamma=0.99, epsilon=epsilon, lr=2.5e-4, alpha=alpha,
                              beta=beta, r_iter=r_iter,
                              input_dims=self.env.observation_space.shape,
                              n_actions=self.env.action_space.n, mem_size=50 * 1024,
                              eps_min=0.01,
                              batch_size=64, replace=replace, eps_dec=1e-4,
                              chkpt_dir=self.model_path + "/", algo='DQNAgent_ranked',
                              env_name=env_id)
        self.myCanvas = Q_NN_canvas(None, network=self.agent.q_eval, bg="white", height=1000, width=1000)
        if self.load_checkpoint:
            self.agent.load_models()
        # self.file_name = self.agent.algo + '_' + self.agent.env_name + '_lr' + str(self.agent.lr) + '_' \
        #                  + str(n_games) + 'games' + str(alpha) + \
        #                  'alpha_' + str(beta) + '_replace_' + str(replace)
        # self.figure_file = 'plots/' + self.file_name + '.png'

        # if you want to record video of your agent playing,
        # do a mkdir tmp && mkdir tmp/dqn-video
        # and uncomment the following 2 lines.
        # env = wrappers.Monitor(env, "tmp/dqn-video",
        #                    video_callable=lambda episode_id: True, force=True)
        self.n_steps = 0
        self.scores, self.eps_history, self.steps_array, self.apples, self.frames = [], [], [], [], []

        self.t1 = time()
        self.best_apple = 0
        self.best_frames = 0
        self.average_frames = 0
        self.average_score = 0
        # self.max
        self.current_generation = 0
        self.population = Pop(self.agent)
        self.done = False
        self.score = 0
        self.info = {"apple score": 0}
        self.frames_per_game = 0
        self.observation = self.env.reset()

    def update(self,display=False):
        if self.done:
            self.scores.append(self.score)
            self.steps_array.append(self.n_steps)
            self.apples.append(self.info["apple score"])
            # print(info["apple score"])
            self.frames.append(self.frames_per_game)
            if self.testing:
                self.test_scores.append(len(self.env.snake_position))

            avg_score = np.mean(self.scores[-100:])
            if self.load_checkpoint:
                print(
                    f"episode: {self.current_generation}, apple: {self.info['apple score']} score: {len(self.env.snake_position)}, frames: {self.frames_per_game}")
            if self.current_generation % self.N_Print == 0:
                save_stats(self.env, self.model_path, "snake_stats")
                t = (time() - self.t1) / self.n_steps * (self.max_frames - self.n_steps) / 60. / 60.
                print('episode: ', self.current_generation, "steps:", self.n_steps, 'score: ', self.score,
                      ' average score %.1f' % avg_score, 'best score %.2f' % self.best_score,
                      'epsilon %.2f' % self.agent.epsilon,
                      f"time: {time() - self.t1}, remaining time: {t} h, apple: {self.best_apple}, frames:{self.best_frames}")
            if self.current_generation % self.N_Print == 0:
                self.agent.save_models(generation=self.current_generation, score=len(self.env.snake_position))
            if avg_score > self.best_score:
                if not self.load_checkpoint and self.agent.epsilon < 0.05:
                    self.agent.save_models(generation=self.current_generation, score=len(self.env.snake_position))
                self.best_score = avg_score
                self.best_apple = np.mean(self.apples[-100:])
                self.best_frames = np.mean(self.frames[-100:])

            self.eps_history.append(self.agent.epsilon)
            # if self.n_steps > self.max_frames:
            #     return
            self.current_generation += 1

            self.observation = self.env.reset()

            self.score = 0
            self.info = {"apple score": 0}
            self.frames_per_game = 0
            self.done = False

        else:

            action = self.agent.choose_action(self.observation)
            # print(action)
            observation_, reward, self.done, info = self.env.step(action)
            self.agent.score = len(self.env.snake_position)
            self.score += reward
            if self.load_checkpoint:
                # 0 up, 1 down, 2 left, 3 right
                # try:
                #     print("action", env.possible_directions[action])
                # except Exception:
                #     print("action",action)
                # print("state", observation_.shape)
                # print("reward", reward)
                # print()
                pass
            #
            # r = clip_reward(reward)
            # print(np.sum(observation), r)
            # print(observation_.shape)
            if not self.load_checkpoint:
                self.agent.store_transition(self.observation, action,
                                            reward, observation_, self.done)
                self.agent.learn()
                # print("learning")
                if display:
                    self.myCanvas.update_network(observation_)
                # self.env.render(mode="human")

                # sleep(0.5)
            self.observation = observation_
            self.n_steps += 1
            self.frames_per_game += 1
            # print(n_steps)


if __name__ == '__main__':
    env = DQNTrainingEnv(None)

    while env.current_generation < env.n_games:
        env.update()
