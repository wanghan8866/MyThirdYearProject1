import gym
import flappy_bird_gym
from meta.appMeta import AppMeta
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

class EnvCreator:

    @staticmethod
    def createEnv(name: str):
        env = None
        if name == "FlappyBird":
            return flappy_bird_gym.make("FlappyBird-rgb-v0")
        game_name = None
        if name == "Breakout":
            game_name = "ALE/Breakout-v5"
            # env = make_atari_env(lambda: gym.make(game_name, render_mode='rgb_array'), n_envs=1, seed=0)
            # # Stack 4 frames
            # env = VecFrameStack(env, n_stack=4)
            # return env

        elif name == "BattleZone":
            game_name = "ALE/BattleZone-v5"
        elif name == "BankHeist":
            game_name = "ALE/BankHeist-v5"
        elif name == "Tennis":
            game_name = "ALE/Tennis-v5"
        elif name == "Skiing":
            game_name = "ALE/Skiing-v5"

        return gym.make(game_name, render_mode='rgb_array')

    @staticmethod
    def returnKeyAction(name: str, key: str):

        if name in AppMeta.GAME_KEY_MAP:
            action, value = AppMeta.KEY_MAP[key]
            if action in AppMeta.GAME_KEY_MAP[name]:
                return AppMeta.GAME_KEY_MAP[name][action]
            else:
                return 0
        return AppMeta.KEY_MAP[key][1]

