from utility.baseAgent import BaseAgent, HumanAgent, BreakoutDDQN, BirdDDQN, SnakeRandomAgent, SnakeGeneticAgent, \
    AStarSnakeAgent, HumanSnakeAgent, DeepQSnakeAgent


class AgentCreator:
    @staticmethod
    def createAgent(name: str, game_name: str, env):

        if game_name == "Snake-gen":
            if name == "randomAgent":
                return SnakeRandomAgent(game_name, env)
            elif name == "A* agent":
                return AStarSnakeAgent(game_name, env)
            elif name == "genetic Agent":
                return SnakeGeneticAgent(game_name, env)
            elif name == "human":
                return HumanSnakeAgent(game_name, env)
            elif name == "DeepQLearningSnakeAgent":
                return DeepQSnakeAgent(game_name, env)

        if name == "randomAgent":
            return BaseAgent(game_name, env)
        if name == "DuelingDDQN" and game_name == "Breakout":
            return BreakoutDDQN(game_name, env)
        if name == "DuelingDDQN" and game_name == "FlappyBird":
            return BirdDDQN(game_name, env)

        return HumanAgent(game_name, env)
