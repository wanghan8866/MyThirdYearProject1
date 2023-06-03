from time import time

import numpy as np

from agents.duelingDDQN.dueling_ddpn_agent import DuelingDDQNAgent
from agents.duelingDDQN.util import make_env, plot_learning_curve


def train(testing=False, n_game=1, n_game_print=20, lr=1e-4, eps_min=0.1, gamma=0.99,
          bath_size=32, replace=1000, eps_dec=1e-5, env_name="BreakoutNoFrameskip-v4",
          max_mem=50000, skip=4, total_frames=1e6):
    t1 = time()
    env = make_env(env_name, repeat=skip)
    best_score = -np.inf
    load_checkpoint = testing
    n_games = n_game
    N = n_game_print
    total_frames = total_frames
    frame_to_save = 0.9 * total_frames
    if load_checkpoint:
        epsilon = 0.
        epsilon_min = 0.
    else:
        epsilon = 1.
        epsilon_min = eps_min
    agent = DuelingDDQNAgent(gamma=gamma, epsilon=epsilon, lr=lr,
                             input_dims=(env.observation_space.shape),
                             n_actions=env.action_space.n, mem_size=max_mem,
                             eps_min=epsilon_min, batch_size=bath_size, replace=replace,
                             eps_dec=eps_dec, chkpt_dir="models/", algo="duelingDDQNAgent",
                             env_name=env_name
                             )
    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + "_" + agent.env_name + "_lr" + str(agent.lr) + "_" + str(n_games) + "games"
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = np.zeros(shape=(n_games,)), np.zeros(shape=(n_games,)), np.zeros(
        shape=(n_games,))
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_state, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, new_state, int(done))
                agent.learn()
            else:
                array = env.render(mode="rgb_array")
            observation = new_state
            n_steps += 1
        scores[i] = score
        steps_array[i] = n_steps
        if i % N == 0:
            avg_score = np.mean(scores[max(0, i - N + 1):(i + 1)])
            print(
                f"episode: {i} score: {score} average score: {avg_score:.1f} epsilon: {agent.epsilon:.2f}, steps:{n_steps},  fps: {(float(time() - t1) / (i + 1))}")
            if avg_score > best_score and n_steps > frame_to_save:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score
        eps_history[i] = agent.epsilon
        if n_steps > total_frames:
            print(i)
            n_games = i + 1
            break

    x = [i + 1 for i in range(n_games)]

    plot_learning_curve(x, scores[:n_games], eps_history[:n_games], figure_file)
    print(f"best result: {best_score}")
    print(f"Time taken: {time() - t1}s")


if __name__ == '__main__':
    train(testing=True, n_game=10, )
