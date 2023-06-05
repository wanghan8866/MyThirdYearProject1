import gym

from agents.PER.ranked.snake_env2 import SnakeEnv2
from agents.duelingDDQN.util import make_env
from agents.snake_game_gen.patterns import Pattern
from agents.snake_game_gen.settings import settings
from agents.snake_game_gen.snake_env3 import load_snake
from meta.appMeta import AppMeta


class EnvCreator:

    @staticmethod
    def createEnv(name: str, agent_name: str = None, pattern=None):
        env = None
        if name == "FlappyBird":
            return make_env("FlappyBird")
        game_name = None
        if name == "Breakout":
            game_name = "BreakoutNoFrameskip-v4"

            return make_env(env_name=game_name)

        elif name == "Qbert":
            game_name = "QbertNoFrameskip-v4"
        elif name == "BankHeist":
            game_name = "ALE/BankHeist-v5"
        elif name == "Tennis":
            game_name = "ALE/Tennis-v5"
        elif agent_name == "DeepQLearningSnakeAgent" and name == "Snake-gen":

            env = SnakeEnv2(10)
            if pattern is not None:
                print("use pattern")
                env.use_pattern(Pattern(pattern))
            return env
        elif name == "Snake-gen":
            env = load_snake("agents/snake_game_gen/models/test_64", f"snake_1699", settings, )
            if pattern is not None:
                print("use pattern")
                env.use_pattern(Pattern(pattern))
            return env

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
