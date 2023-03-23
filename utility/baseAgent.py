from utility.envCreator import EnvCreator
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import os, sys
import gym
from agents.duelingDDQN.util import make_env
from agents.duelingDDQN.dueling_ddpn_agent import DuelingDDQNAgent
from snake_game_gen.snake_env3 import Snake
from snake_game_gen.path_finding import Mixed
from snake_game_gen.misc import *
from snake_game_gen.tk_nn import NN_canvas
from PER.ranked.agent import DQNAgent
from PER.ranked.deepQ_nn_vis import Q_NN_canvas

class BaseAgent:
    def __init__(self, game_name, env):
        self.game_name = game_name
        self.env = env

    def predict(self, state, action_space, action):
        return action_space.sample()

    def render(self):
        return self.env.render(mode="rgb_array")


class HumanAgent(BaseAgent):
    def predict(self, state, action_space, action):
        if action:
            return action
        return 0


class BreakoutDDQN(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        # print(os.path.curdir)
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))
        # env = make_env(env_name, repeat=skip)
        self.agent = DuelingDDQNAgent(gamma=0.99, epsilon=0, lr=1e-4,
                                      input_dims=(self.env.observation_space.shape),
                                      n_actions=self.env.action_space.n, mem_size=50000,
                                      eps_min=0.1, batch_size=32, replace=1000,
                                      eps_dec=1e-5, chkpt_dir="resources/models", algo="duelingDDQNAgent",
                                      env_name="BreakoutNoFrameskip-v4"
                                      )
        self.agent.load_models()
        # self.myCanvas = NN_canvas(None, network=self.agent.q_eval, bg="white", height=1000, width=1000)

    def predict(self, state, action_space, action):
        return self.agent.choose_action(state)


class BirdDDQN(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        # print(os.path.curdir)
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))
        # env = make_env(env_name, repeat=skip)
        # print(self.env.observation_space.shape)
        self.agent = DuelingDDQNAgent(gamma=0.99, epsilon=0, lr=1e-4,
                                      input_dims=(self.env.observation_space.shape),
                                      n_actions=self.env.action_space.n, mem_size=50000,
                                      eps_min=0.1, batch_size=32, replace=1000,
                                      eps_dec=1e-5, chkpt_dir="resources/models", algo="duelingDDQNAgent",
                                      env_name="FlappyBird"
                                      )
        self.agent.load_models()

    def predict(self, state, action_space, action):
        return self.agent.choose_action(state)


class SnakeRandomAgent(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        self.agent = env

    def predict(self, state, action_space, action):
        print("predicted action", self.agent.possible_directions[self.agent.action_space.sample()])
        return self.agent.possible_directions[self.agent.action_space.sample()]


class SnakeGeneticAgent(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        self.agent = env
        self.nn = NN_canvas(None, snake=env, bg="white", height=1000, width=1000)

    def predict(self, state, action_space, action):
        # print("predicted action", self.agent.possible_directions[self.agent.action_space.sample()] )
        return -1
    def render(self):
        # self.nn.update_network()
        return self.env.render(mode="rgb_array")


class AStarSnakeAgent(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        self.agent = env
        self.path=[]

    def predict(self, state, action_space, action):
        # print("predicted action", self.agent.possible_directions[self.agent.action_space.sample()] )
        path = Mixed(self.agent, self.agent.apple_location).run_mixed()
        self.path =path
        # print("path: ", path)
        if path is None:
            action = -1
        elif path[1] is None:
            action = -1
        else:
            # print("path: ", *path)

            result = path[1] - self.agent.snake_array[0]
            old_action = action
            if result == Point(0, 1):
                action = "d"
            elif result == Point(0, -1):
                action = "u"
            elif result == Point(1, 0):
                action = "r"
            else:
                action = "l"
            # if old_action == action:
            #     path_correctness[episode].append(1)
            # else:
            #     path_correctness[episode].append(0)
        return action

    def render(self):
        return self.env.render(mode="rgb_array", drawing_vision=False, path=self.path)


class HumanSnakeAgent(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        self.agent = env
        self.old_action = -1

    def predict(self, state, action_space, action):
        # print("predicted action", self.agent.possible_directions[self.agent.action_space.sample()] )
        # path = Mixed(self.agent, self.agent.apple_location).run_mixed()
        # print("human", action)
        if action == 2:
            self.old_action = "u"
            # return "u"
        elif action == 3:
            self.old_action = "r"
            # return "r"
        elif action == 4:
            self.old_action = "l"
            # return "l"
        elif action == 5:
            self.old_action = "d"
            # return "d"
        return self.old_action

    def render(self):
        return self.env.render(mode="rgb_array", drawing_vision=False)



class DeepQSnakeAgent(BaseAgent):
    def __init__(self, game_name, env):
        super().__init__(game_name, env)
        alpha = 0.25
        beta = 0.5
        r_iter = 64
        env_id = "snake-v2"
        replace = 250
        epsilon = 0
        self.agent = DQNAgent(gamma=0.99, epsilon=epsilon, lr=2.5e-4, alpha=alpha,
                     beta=beta, r_iter=r_iter,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50 * 1024,
                     eps_min=0.01,
                     batch_size=64, replace=replace, eps_dec=1e-4,
                     chkpt_dir='PER/ranked/models/', algo='DQNAgent_ranked',
                     env_name=env_id)
        self.agent.load_models()
        self.myCanvas = Q_NN_canvas(None, network=self.agent.q_eval, bg="white", height=1000, width=1000)

    def predict(self, state, action_space, action):
        # print("predicted action", self.agent.possible_directions[self.agent.action_space.sample()] )
        # path = Mixed(self.agent, self.agent.apple_location).run_mixed()
            # return "d"
        # self.myCanvas.update_network(state)
        return self.agent.choose_action(state)

    def render(self):
        return self.env.render(mode="rgb_array")



if __name__ == '__main__':
    print(os.path.curdir)
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    # A2C.load("resources/agents/Breakout")
    # env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
    # Stack 4 frames
    # env = VecFrameStack(env, n_stack=4)
    # model = A2C('CnnPolicy', env, verbose=1,
    #             ent_coef=0.01,
    #             vf_coef=0.25,
    #             policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
    #             )
    env = make_atari_env('ALE/Breakout-v5', n_envs=8, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    print(env.observation_space.shape)
    print(env.action_space)
    # model = A2C('CnnPolicy', env, verbose=1,
    #             ent_coef=0.01,
    #             vf_coef=0.25,
    #             policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
    #             )
    # model.learn(total_timesteps=int(1e4))
    # model.save("Breakout_w")
    #
    # del model
    #
    env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    print(env.observation_space.shape)
    print(env.action_space)
    # obs = env.reset()
    # new_model = A2C.load("../resources/agents/Breakout1.zip", env=env)
    # print(new_model.predict(obs))
