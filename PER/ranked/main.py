import gym
import numpy as np
from agent import DQNAgent
from utils import plot_learning_curve
from snake_env4 import Snake
from snake_env2 import SnakeEnv2
# from snake_env5 import SnakeEnv5
from time import time, sleep
import cv2
import collections
from deepQ_nn_vis import NN_canvas


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
        self.observation_space = gym.spaces.Box(
            np.array(env.observation_space.low).repeat(repeat, axis=0),
            np.array(env.observation_space.high).repeat(repeat, axis=0),

            dtype=np.uint8)
        # print(self.observation_space.low.shape)
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


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env_id = "snake-v2-r16"
    # env = Snake([10,10])
    env = SnakeEnv2(10)
    print(env.observation_space.shape)
    env = StackFrames(env, repeat=16)
    print(env.observation_space.shape)
    best_score = -np.inf
    load_checkpoint = True
    testing = True
    n_games = int(1e6)
    max_frames = int(1e6)
    game_to_save = 0.5 * max_frames
    r_iter = 64
    alpha = 0.25
    beta = 0.5
    replace = 250
    N_Print = 300
    if testing:
        n_games = 100
        test_scores=[]

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
    myCanvas = NN_canvas(None, network=agent.q_eval, bg="white", height=1000, width=1000)
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
            # print(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            if load_checkpoint:
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
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()

            elif not testing:
                t_end = time() + 0.1
                k = -1
                # action = -1
                while time() < t_end:
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

                env.render(mode="human")
                myCanvas.update_network(observation_)
                # sleep(0.5)
            observation = observation_
            n_steps += 1
            frames_per_game += 1
            # print(n_steps)
        scores.append(score)
        steps_array.append(n_steps)
        apples.append(info["apple score"])
        # print(info["apple score"])
        frames.append(frames_per_game)
        if testing:
            test_scores.append(len(env.snake_position))

        avg_score = np.mean(scores[-100:])
        if load_checkpoint:
            print(
                f"episode: {i}, apple: {info['apple score']} score: {len(env.snake_position)}, frames: {frames_per_game}")
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

    if testing:
        print(f"After {n_games}, the average score : {np.mean(test_scores)}")
    if not load_checkpoint:
        x = [i + 1 for i in range(len(scores))]
        np.save(f"models/{env_id}_DQNAgent_x", x)
        np.save(f"models/{env_id}_DQNAgent_scores", scores)
        plot_learning_curve(steps_array, scores, eps_history, figure_file)
