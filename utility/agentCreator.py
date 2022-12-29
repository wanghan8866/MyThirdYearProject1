from utility.baseAgent import BaseAgent,HumanAgent, BreakoutA2C


class AgentCreator:
    @staticmethod
    def createAgent(name: str, game_name:str):
        if name == "randomAgent":
            return BaseAgent(game_name)
        if name=="breakoutA2C":
            return BreakoutA2C(game_name)

        return HumanAgent(game_name)
