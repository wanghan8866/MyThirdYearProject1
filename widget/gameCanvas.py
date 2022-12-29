import gym
from gym import Env
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from utility.envCreator import EnvCreator
from utility.agentCreator import AgentCreator
from typing import List
from meta.appMeta import AppMeta

print()
dones = False
MOVE_INCREMENT = 20
MOVES_PER_SECOND = 15
GAME_SPEED = 1000 // MOVES_PER_SECOND


class GameCanvas(ctk.CTkCanvas):
    layouts = [[(0, 0)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (0.5, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)]]

    def __init__(self, container, game_name: str, agent_name: str, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        # if len(game_names) != len(agent_names):
        #     raise ValueError("length of games not matching!")
        # self.N = len(game_names)
        # self.configure(width=1000, height=1000, background="yellow")
        # self.envs: List[Env] = [EnvCreator.createEnv(game) for game in game_names]
        self.env: Env = EnvCreator.createEnv(game_name)

        # obs_list = [env.reset() for env in self.envs]
        self.obs = self.env.reset()
        self.flip = False
        self.game_name = game_name
        self.agent_name = agent_name
        if game_name == "FlappyBird":
            self.flip = True
            self.obs = np.transpose(self.obs, axes=(1, 0, 2))
        print(self.obs.shape)
        print(self.env.action_space)
        if len(self.obs.shape) > 2:
            self.configure(height=self.obs.shape[0], width=self.obs.shape[1])
        self.container = container
        # self.models = [AgentCreator.createAgent(agent_name) for agent_name in agent_names]
        self.agent = AgentCreator.createAgent(agent_name, game_name)
        # self.container.img=None
        # self.dones = np.zeros(shape=(self.N,))
        self.images = [None]
        # self.layout = GameCanvas.layouts[self.N - 1]
        self.current_action = None
        self.total_reward = 0

        self.after(GAME_SPEED, self.move)

    def move(self):
        # self.update()
        # action = self.model.predict(None, self.env.action_space)
        # images = []
        # i = 0

        action = self.agent.predict(self.obs, self.env.action_space, self.current_action)
        # print(action)
        self.obs, rewards, done, info = self.env.step(action)
        self.current_action = None
        self.total_reward += rewards
        array = self.env.render(mode="rgb_array")
        if self.flip:
            array = np.transpose(array, axes=(1, 0, 2))
        self.images[0] = ImageTk.PhotoImage(image=Image.fromarray(array))
        # self.create_image((self.layout[i][0] * obs.shape[1], self.layout[i][1] * obs.shape[0]), anchor="nw",
        #                   image=self.images[0])
        self.create_image((0, 0), anchor="nw",
                          image=self.images[0])

        # self.dones[i] = 1 if done else 0
        # i += 1

        # self.create_text(10, 10, text=f"hell+{self}")
        # plt.imshow(array)
        # plt.show()
        # print(obs)

        if not done:
            # print()
            # print(self, self.env.action_space, np.sum(array))
            self.after(GAME_SPEED, self.move)
        else:
            # [env.close() for env in self.envs]
            self.delete("all")
            self.create_text(int(self.cget("width")) // 2, int(self.cget("height")) // 2,
                             text=f"Game Over with score {self.total_reward}",
                             fill="white")
            self.env.close()
            # self.destroy()
            # self.master.destroy()
            print("done")

    def on_key_press(self, e):
        key = e.keysym
        if key in AppMeta.KEY_MAP:
            self.current_action = EnvCreator.returnKeyAction(self.game_name, key)


if __name__ == '__main__':
    root = ctk.CTk()
    # root.geometry("600x400")

    # canvas = GameCanvas(root, game_names=["Skiing", "Skiing", "BattleZone"],
    #                     agent_names=["randomAgent", "randomAgent", "randomAgent"])
    # canvas.grid(row=0, column=0)
    #
    # canvas = GameCanvas(root, game_names=["Skiing", "Skiing", "BattleZone"],
    #                     agent_names=["randomAgent", "randomAgent", "randomAgent"])
    # canvas.grid(row=1, column=1)
    canvas1 = GameCanvas(root, game_name="Skiing", agent_name="randomAgent", width=300, height=300, background="black")
    canvas1.grid(row=0, column=0)
    canvas1 = GameCanvas(root, game_name="Skiing", agent_name="randomAgent", width=300, height=300, background="black")
    canvas1.grid(row=0, column=1)
    canvas = GameCanvas(root, game_name="BattleZone", agent_name="randomAgent", width=300, height=300,
                        background="black")
    canvas.grid(row=1, column=0)
    # canvas = GameCanvas(root, game_name="Breakout", agent_name="randomAgent", width=300, height=300, background="black")
    # canvas.grid(row=1, column=1)

    root.mainloop()
