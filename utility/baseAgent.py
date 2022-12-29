from utility.envCreator import EnvCreator
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import os, sys
import gym


class BaseAgent:
    def __init__(self, game_name):
        self.game_name = game_name

    def predict(self, state, action_space, action):
        return action_space.sample()


class HumanAgent(BaseAgent):
    def predict(self, state, action_space, action):
        if action:
            return action
        return 0


class BreakoutA2C(BaseAgent):
    def __init__(self, game_name):
        super().__init__(game_name)
        # print(os.path.curdir)
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))
        self.agent = A2C.load("resources/agents/Breakout_test")

    def predict(self, state, action_space, action):
        return self.agent.predict(state)


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
