import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from gym import Env

from meta.appMeta import AppMeta
from utility.agentCreator import AgentCreator
from utility.envCreator import EnvCreator

print()
dones = False
MOVE_INCREMENT = 20
MOVES_PER_SECOND = 10
GAME_SPEED = 1000 // MOVES_PER_SECOND


class GameCanvas(ctk.CTkFrame):
    layouts = [[(0, 0)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (0.5, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)]]

    def __init__(self, container, game_name: str, agent_name: str, pattern=None, background="black", create_env=None,
                 *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        # if len(game_names) != len(agent_names):
        #     raise ValueError("length of games not matching!")
        # self.N = len(game_names)
        # self.configure(width=1000, height=1000, background="yellow")
        # self.envs: List[Env] = [EnvCreator.createEnv(game) for game in game_names]
        if create_env is not None:
            self.env = create_env(pattern=pattern)
        else:
            self.env: Env = EnvCreator.createEnv(game_name, agent_name=agent_name, pattern=pattern)

        # obs_list = [env.reset() for env in self.envs]
        self.obs = self.env.reset()
        array = self.env.render(mode="rgb_array")
        # print(array.shape)
        self.configure(width=array.shape[0], height=array.shape[1])
        self.flip = False
        self.game_name = game_name
        self.agent_name = agent_name

        if game_name == "FlappyBird":
            self.flip = True
            array = np.transpose(array, axes=(1, 0, 2))
        print(array.shape)
        print(self.env.action_space)
        self.speed = GAME_SPEED
        self.speed_label = ctk.CTkLabel(
            self,
            text=f"Game Speed: in ms per frame",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_label.pack(side="top")

        self.speed_entry = ctk.CTkEntry(
            self,
            placeholder_text=f"{self.speed}",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_entry.pack(side="top")

        self.canvas = ctk.CTkCanvas(self, background=background, width=array.shape[1], height=array.shape[0])
        self.canvas.pack(side="bottom", fill="both", expand=True)

        if len(array.shape) > 2:
            self.configure(height=array.shape[0], width=array.shape[1])
        self.container = container
        # self.models = [AgentCreator.createAgent(agent_name) for agent_name in agent_names]
        self.agent = AgentCreator.createAgent(agent_name, game_name, self.env)
        # self.container.img=None
        # self.dones = np.zeros(shape=(self.N,))
        self.images = [None]
        # self.layout = GameCanvas.layouts[self.N - 1]
        self.current_action = None
        self.total_reward = 0

        self.canvas.after(self.speed, self.move)

    def move(self):
        # self.update()
        # action = self.model.predict(None, self.env.action_space)
        # images = []
        # i = 0
        # print("game updating")

        action = self.agent.predict(self.obs, self.env.action_space, self.current_action)
        # print(action)
        self.obs, rewards, done, info = self.env.step(action)
        self.current_action = None
        self.total_reward = rewards
        array = self.agent.render()
        if self.flip:
            array = np.transpose(array, axes=(1, 0, 2))
        self.images[0] = ImageTk.PhotoImage(image=Image.fromarray(array))
        # self.create_image((self.layout[i][0] * obs.shape[1], self.layout[i][1] * obs.shape[0]), anchor="nw",
        #                   image=self.images[0])
        self.canvas.create_image((0, 0), anchor="nw",
                                 image=self.images[0])
        # self.configure(width=)

        # self.dones[i] = 1 if done else 0
        # i += 1

        # self.create_text(10, 10, text=f"hell+{self}")
        # plt.imshow(array)
        # plt.show()
        # print(obs)

        if not done:
            # print()
            # print(self, self.env.action_space, np.sum(array))
            try:
                self.speed = int(self.speed_entry.get())
            except Exception:
                self.speed_entry.delete(0, len(self.speed_entry.get()))
                self.speed_entry.insert(0, self.speed)

            self.canvas.after(self.speed, self.move)
        else:
            # [env.close() for env in self.envs]
            self.canvas.delete("all")
            self.canvas.create_text(int(self.cget("width")) // 2, int(self.cget("height")) // 2,
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
